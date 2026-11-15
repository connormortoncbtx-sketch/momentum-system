"""
automation/shadow_premium.py
============================
Shadow #11 (no orders), approved Nov 15, 2026: the option-premium TRIO -- same-day premium sold at three times of day
on SPY and QQQ (research/quick_trio.py, docs "Option-premium variants"):

  slots     NOON  12:00 ET  put spread   short put one implied move below, long put a width lower
            2PM   14:00 ET  iron condor  short put/call one implied move out, wings a width further
            3PM   15:00 ET  iron condor  same, at 15:00
            implied move = ATM straddle mid at entry; width SPY $5, QQQ ~0.8% of the price (as shadow_condor.py);
            held to the 4pm close (the research's hold and 2x-stop variants are within ~0.2% per trade of each other)
  pricing   NATURAL  sell at the bid, buy at the ask (worst case, live quotes at entry)
            MID      quote midpoints (best case)
  books     per slot: 2% of NAV at risk a day (1% SPY + 1% QQQ)            -> noon_*, 2pm_*, 3pm_*
            TRIO: the same 2% a day split equally across the three slots   -> trio_*
  backtest  Sharpe (daily RoR) 2018-23 / 2024-26: noon put 1.06 / 1.00, 2pm condor 1.25 / 3.25, 3pm condor 2.83 / 2.86;
            equal-risk trio 2.30 / 3.33; correlations with the 2pm 0.28-0.49. All three can max out on the same crash
            day, so the trio's daily risk is sized for a total loss. The live 3pm paper account (k=0.5) is behind its
            backtest -- this shadow measures real entry prices for all three.
State: data/shadow_premium_state.json   Ledger: data/shadow_premium_ledger.jsonl (one row per settled day)
"""
import datetime as dt
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from automation.shadow_condor import ET, clients, quotes, spot_now, official_close, width, occ  # noqa: E402
from automation.system_logger import log_event, LogStatus  # noqa: E402
from automation.notifier import notify_success, notify_alert  # noqa: E402

log = logging.getLogger("shadow_premium")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
DATA = ROOT / "data"
STATE, LEDGER = DATA / "shadow_premium_state.json", DATA / "shadow_premium_ledger.jsonl"
UNDS, K, RISK = ("SPY", "QQQ"), 1.0, 0.02
SLOTS = {"noon": ("put", dt.time(11, 58), dt.time(12, 40)),
         "2pm": ("condor", dt.time(13, 58), dt.time(14, 40)),
         "3pm": ("condor", dt.time(14, 58), dt.time(15, 30))}
PRICES = ("natural", "mid")
NAVS = [f"{s}_{p}" for s in SLOTS for p in PRICES] + [f"trio_{p}" for p in PRICES]


def enter_one(oc, sc, und, day, now, struct):
    s0, sfeed = spot_now(sc, und)
    atm, W = round(s0), width(und, s0)
    q1, _ = quotes(oc, [occ(und, day, "C", atm), occ(und, day, "P", atm)])
    mid = lambda q: (q["bid"] + q["ask"]) / 2
    M = mid(q1[occ(und, day, "C", atm)]) + mid(q1[occ(und, day, "P", atm)])
    ps, cs = round(s0 - K * M), round(s0 + K * M)
    legs = {"short_put": occ(und, day, "P", ps), "long_put": occ(und, day, "P", ps - W)}
    if struct == "condor":
        legs.update({"short_call": occ(und, day, "C", cs), "long_call": occ(und, day, "C", cs + W)})
    q, feed = quotes(oc, list(legs.values()))
    Q = {k: q[v] for k, v in legs.items()}
    natural = sum(Q[k]["bid"] if k.startswith("short") else -Q[k]["ask"] for k in Q)
    midc = sum((1 if k.startswith("short") else -1) * mid(Q[k]) for k in Q)
    log.info(f"entered {struct} {und} {day} at {now:%H:%M}: {s0:.2f}, move {M:.2f}, puts {ps}/{ps - W}"
             + (f", calls {cs}/{cs + W}" if struct == "condor" else "") + f", credit natural {natural:.2f} / mid {midc:.2f} ({feed})")
    return {"und": und, "struct": struct, "entered_at": now.isoformat(timespec="seconds"), "spot": s0, "quote_feed": feed,
            "implied_move": round(M, 4), "width": W, "put_strikes": [ps, ps - W],
            "call_strikes": [cs, cs + W] if struct == "condor" else None, "legs": legs, "quotes": Q,
            "credit": {"natural": round(natural, 4), "mid": round(midc, 4)}}


def settle_one(sc, e, day):
    sT = official_close(sc, e["und"], day)
    if sT is None:
        return None
    W, ps = e["width"], e["put_strikes"][0]
    loss = min(max(ps - sT, 0), W)
    if e["struct"] == "condor":
        loss += min(max(sT - e["call_strikes"][0], 0), W)
    out = {}
    for p in PRICES:
        c = e["credit"][p]
        out[p] = None if (c is None or c <= 0 or c >= W) else round((c - loss) / (W - c), 4)
    return {"und": e["und"], "struct": e["struct"], "close": sT, "move_pct": round((sT / e["spot"] - 1) * 100, 3),
            "credit": e["credit"], "loss_per_share": round(loss, 4), "ror": out}


def main():
    st = json.loads(STATE.read_text()) if STATE.exists() else {"nav": {}, "open": {}, "done_days": []}
    for b in NAVS:
        st["nav"].setdefault(b, 100000.0)
    now = dt.datetime.now(ET)
    oc, sc, tc = clients()
    # settle earlier days
    rows = []
    for key in sorted(st["open"]):
        day = dt.date.fromisoformat(key)
        if day >= now.date():
            continue
        ent = st["open"][key]
        res = {slot: {u: settle_one(sc, e, day) for u, e in d.items()} for slot, d in ent.items()}
        if any(v is None for d in res.values() for v in d.values()):
            log.info(f"{key}: close not available yet"); continue
        nav = st["nav"]
        for p in PRICES:
            slot_r = {}
            for slot, d in res.items():
                rs = [x["ror"][p] for x in d.values() if x["ror"].get(p) is not None]
                if rs:
                    slot_r[slot] = sum(rs) / len(rs)                     # 1% SPY + 1% QQQ = 2% on the slot's average
                    nav[f"{slot}_{p}"] *= 1 + RISK * slot_r[slot]
            if slot_r:                                                   # trio: 2% split over the slots that traded
                nav[f"trio_{p}"] *= 1 + RISK / len(SLOTS) * sum(slot_r.values())
        rows.append({"date": key, "slots": res, "nav": {b: round(nav[b], 2) for b in NAVS}})
        st["done_days"].append(key)
        del st["open"][key]
    # enter any slot whose window is open now
    key = now.date().isoformat()
    if key not in st["done_days"] and now.weekday() < 5:
        for slot, (struct, t0, t1) in SLOTS.items():
            if not (t0 <= now.time() <= t1):
                continue
            have = st["open"].setdefault(key, {}).setdefault(slot, {})
            todo = [u for u in UNDS if u not in have]
            if not todo:
                continue
            if not tc.get_clock().is_open:
                log.info("market closed -- no entry"); break
            for und in todo:
                try:
                    have[und] = enter_one(oc, sc, und, now.date(), now, struct)
                except Exception as ex:
                    log.error(f"{slot} {und} entry failed: {ex}")
                    log_event("shadow_premium", LogStatus.ERROR, f"{slot} {und} entry failed: {ex}")
                    notify_alert("Shadow premium trio", f"{slot} {und} entry failed: {ex}")
            if not have:
                st["open"][key].pop(slot, None)
        if key in st["open"] and not st["open"][key]:
            st["open"].pop(key)
    STATE.write_text(json.dumps(st, indent=1))
    if rows:
        with LEDGER.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        for r in rows:
            n = r["nav"]
            parts = [f"{slot}: " + ", ".join(f"{u} {x['ror']['natural'] * 100:+.0f}%" if x["ror"]["natural"] is not None else f"{u} n/a"
                                             for u, x in d.items()) for slot, d in r["slots"].items()]
            msg = (f"{r['date']}: " + "; ".join(parts) + f". NAV natural: trio {n['trio_natural']:,.0f}, noon {n['noon_natural']:,.0f}, "
                   f"2pm {n['2pm_natural']:,.0f}, 3pm {n['3pm_natural']:,.0f}")
            log.info(msg)
            log_event("shadow_premium", LogStatus.SUCCESS, msg, metrics={k: n[k] for k in NAVS})
            notify_success("Shadow premium trio", msg)


if __name__ == "__main__":
    main()
