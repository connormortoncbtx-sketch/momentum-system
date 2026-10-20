"""
automation/shadow_condor.py
===========================
Shadow #9 (no orders): same-day iron condors on SPY and QQQ (research/options_premium.py, Oct 20, 2026).

  setup     each trading day at ~10:00 ET, an iron condor expiring today: short call and short put one implied move
            (ATM straddle mid) from the price, insurance further out -- SPY $5 (the SPY research grid), QQQ ~0.8% of
            the price (the ETF test's width; SPY's $5 was ~0.8% when chosen); held to the 4pm close
  books     per pricing: NATURAL  sell at the bid, buy at the ask (worst case, live quotes at entry)
                         MID      quote midpoints (best case)
                         VWAP1C   the backtest's price: 10:00-10:15 trade VWAP per leg, $0.01 per leg cost
            per sizing:  SPY      2% of NAV at risk on SPY (book names natural / mid / vwap1c, from Oct 20)
                         QQQ      2% on QQQ (qqq_*)
                         COMBO    1% SPY + 1% QQQ (combo_*): same total risk, the QQQ test's proposal
  backtest  SPY 0DTE condor, 1x implied move, $5 wings, $0.01/leg: RoR +3.1%/trade both halves (Sharpe 1.55 -> 1.69),
            at $0.03/leg +1.2% (t 1.0). QQQ out of sample: +6.0% -> +4.5%/trade (t 2.9 / 2.4); daily P&L
            correlation with SPY 0.86. The cost of a real fill decides it -- NATURAL vs MID vs VWAP1C measure that.

It places no orders (Alpaca paper fills are optimistic, so paper orders would not answer the cost question).
Quotes come from the OPRA feed when the account has it, else Alpaca's indicative feed (logged per entry).
Each run: settle earlier days' entries (next morning: without a signed OPRA agreement Alpaca refuses same-day option
bars, which VWAP1C needs), then enter today for each underlying not yet entered, inside 09:58-11:30 ET.
State: data/shadow_condor_state.json   Ledger: data/shadow_condor_ledger.jsonl (one row per settled day)
"""
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from automation.system_logger import log_event, LogStatus  # noqa: E402
from automation.notifier import notify_success, notify_alert  # noqa: E402

log = logging.getLogger("shadow_condor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
ET = ZoneInfo("America/New_York")
DATA = ROOT / "data"
STATE, LEDGER = DATA / "shadow_condor_state.json", DATA / "shadow_condor_ledger.jsonl"
UNDS, K, RISK_FRAC = ("SPY", "QQQ"), 1.0, 0.02
ENTRY_FROM, ENTRY_TO = dt.time(9, 58), dt.time(11, 30)
PRICES = ("natural", "mid", "vwap1c")
NAVS = list(PRICES) + [f"qqq_{p}" for p in PRICES] + [f"combo_{p}" for p in PRICES]


def width(und, s0):
    return 5 if und == "SPY" else max(1, round(0.008 * s0))


def occ(und, exp, cp, k):
    return f"{und}{exp:%y%m%d}{cp}{int(round(k * 1000)):08d}"


def clients():
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    k, s = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    return OptionHistoricalDataClient(k, s), StockHistoricalDataClient(k, s), TradingClient(k, s, paper=True)


def quotes(oc, syms):
    """Latest bid/ask per option symbol; OPRA if the account has it, else indicative."""
    from alpaca.data.requests import OptionLatestQuoteRequest
    from alpaca.data.enums import OptionsFeed
    last = None
    for feed in (OptionsFeed.OPRA, OptionsFeed.INDICATIVE):
        try:
            q = oc.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=syms, feed=feed))
            out = {s: {"bid": float(v.bid_price), "ask": float(v.ask_price), "t": v.timestamp.isoformat()}
                   for s, v in q.items()}
            if all(s in out and out[s]["ask"] > 0 for s in syms):
                return out, feed.value
            last = f"{feed.value}: missing {[s for s in syms if s not in out]}"
        except Exception as e:
            last = f"{feed.value}: {e}"
    raise RuntimeError(f"no option quotes ({last})")


def spot_now(sc, und):
    from alpaca.data.requests import StockLatestTradeRequest
    from alpaca.data.enums import DataFeed
    for feed in (DataFeed.SIP, DataFeed.IEX):
        try:
            t = sc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=und, feed=feed))[und]
            return float(t.price), feed.value
        except Exception as e:
            log.info(f"{und} spot via {feed.value}: {e}")
    raise RuntimeError(f"no {und} price")


def official_close(sc, und, day):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    end = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20)
    df = sc.get_stock_bars(StockBarsRequest(symbol_or_symbols=und, timeframe=TimeFrame.Day,
                                            start=dt.datetime.combine(day, dt.time()), end=end,
                                            adjustment=Adjustment.RAW, feed=DataFeed.SIP)).df.reset_index()
    df["date"] = df.timestamp.dt.tz_convert(ET).dt.date
    r = df[df.date == day]
    return float(r.close.iloc[0]) if len(r) else None


def vwap_prices(oc, day, syms):
    """10:00-10:15 ET trade VWAP per leg -- the backtest's entry price."""
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    st = dt.datetime.combine(day, dt.time(10, 0), tzinfo=ET)
    df = oc.get_option_bars(OptionBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                                              start=st, end=st + dt.timedelta(minutes=15))).df
    out = {}
    if len(df):
        df = df.reset_index()
        for s, g in df.groupby("symbol"):
            out[s] = float((g.vwap * g.volume).sum() / g.volume.sum()) if g.volume.sum() > 0 else float(g.close.mean())
    return out


def enter_one(oc, sc, und, day, now):
    s0, sfeed = spot_now(sc, und)
    atm, W = round(s0), width(und, s0)
    q1, _ = quotes(oc, [occ(und, day, "C", atm), occ(und, day, "P", atm)])
    mid = lambda q: (q["bid"] + q["ask"]) / 2
    M = mid(q1[occ(und, day, "C", atm)]) + mid(q1[occ(und, day, "P", atm)])
    ps, cs = round(s0 - K * M), round(s0 + K * M)
    legs = {"short_put": occ(und, day, "P", ps), "long_put": occ(und, day, "P", ps - W),
            "short_call": occ(und, day, "C", cs), "long_call": occ(und, day, "C", cs + W)}
    q, feed = quotes(oc, list(legs.values()))
    Q = {k: q[v] for k, v in legs.items()}
    natural = Q["short_put"]["bid"] + Q["short_call"]["bid"] - Q["long_put"]["ask"] - Q["long_call"]["ask"]
    midc = sum((1 if k.startswith("short") else -1) * mid(Q[k]) for k in Q)
    log.info(f"entered {und} {day}: {s0:.2f}, move {M:.2f}, puts {ps}/{ps - W}, calls {cs}/{cs + W}, "
             f"credit natural {natural:.2f} / mid {midc:.2f} ({feed})")
    return {"und": und, "date": day.isoformat(), "entered_at": now.isoformat(timespec="seconds"), "spot": s0,
            "spot_feed": sfeed, "quote_feed": feed, "implied_move": round(M, 4), "width": W,
            "put_strikes": [ps, ps - W], "call_strikes": [cs, cs + W], "legs": legs, "quotes": Q,
            "credit": {"natural": round(natural, 4), "mid": round(midc, 4)},
            "leg_spread_avg": round(sum(Q[k]["ask"] - Q[k]["bid"] for k in Q) / 4, 4)}


def enter(state, oc, sc, tc, now):
    day, key = now.date(), now.date().isoformat()
    if key in state["done_days"] or not (ENTRY_FROM <= now.time() <= ENTRY_TO):
        if now.time() > ENTRY_TO and now.weekday() < 5 and len(state["open"].get(key, {})) < len(UNDS):
            log.info("entry window passed without a full entry today")
        return
    have = state["open"].setdefault(key, {})
    todo = [u for u in UNDS if u not in have]
    if not todo:
        return
    if not tc.get_clock().is_open:
        log.info("market closed today -- no entry")
        if not have:
            state["open"].pop(key, None)
        return
    for und in todo:
        try:
            have[und] = enter_one(oc, sc, und, day, now)
        except Exception as e:
            log.error(f"{und} entry failed: {e}")
            log_event("shadow_condor", LogStatus.ERROR, f"{und} entry failed: {e}")
            notify_alert("Shadow condor", f"{und} entry failed: {e}")
    if not have:
        state["open"].pop(key, None)


def settle_one(oc, sc, e, day):
    """Return-on-risk per pricing for one entry, or None if the close isn't available yet."""
    und, W = e.get("und", "SPY"), e.get("width", 5)
    sT = official_close(sc, und, day)
    if sT is None:
        return None
    ps, cs = e["put_strikes"][0], e["call_strikes"][0]
    loss = min(max(ps - sT, 0), W) + min(max(sT - cs, 0), W)
    credits = dict(e["credit"])
    try:
        v = vwap_prices(oc, day, list(e["legs"].values()))
        px = {k: v.get(s, 0.01 if k.startswith("long") else None) for k, s in e["legs"].items()}   # untraded wing: 1c
        if None not in px.values():
            credits["vwap1c"] = round(px["short_put"] + px["short_call"] - px["long_put"] - px["long_call"] - 4 * 0.01, 4)
    except Exception as ex:
        log.warning(f"{und} {day}: vwap prices: {ex}")
    res = {}
    for p in PRICES:
        c = credits.get(p)
        res[p] = None if (c is None or c <= 0 or c >= W) else \
            {"credit": c, "pnl_per_share": round(c - loss, 4), "return_on_risk": round((c - loss) / (W - c), 4)}
    return {"und": und, "close": sT, "move_pct": round((sT / e["spot"] - 1) * 100, 3), "implied_move": e["implied_move"],
            "put_strikes": e["put_strikes"], "call_strikes": e["call_strikes"], "loss_per_share": round(loss, 4),
            "quote_feed": e["quote_feed"], "leg_spread_avg": e["leg_spread_avg"], "entered_at": e["entered_at"],
            "books": res}


def settle(state, oc, sc, now):
    rows = []
    for key in sorted(state["open"]):
        day = dt.date.fromisoformat(key)
        if day >= now.date():
            continue
        ent = state["open"][key]
        out = {u: settle_one(oc, sc, e, day) for u, e in ent.items()}
        if any(v is None for v in out.values()):
            log.info(f"{key}: close not available yet"); continue
        nav = state["nav"]
        ror = lambda u, p: (out.get(u) or {}).get("books", {}).get(p) and out[u]["books"][p]["return_on_risk"]
        for p in PRICES:
            s, q = ror("SPY", p), ror("QQQ", p)
            if s is not None:
                nav[p] *= 1 + RISK_FRAC * s
            if q is not None:
                nav[f"qqq_{p}"] *= 1 + RISK_FRAC * q
            parts = [x for x in (s, q) if x is not None]
            if parts:                                  # 1% each; if one side had no valid trade, the other at 1%
                nav[f"combo_{p}"] *= 1 + RISK_FRAC / 2 * sum(parts)
        rows.append({"date": key, "legs": out, "nav": {b: round(nav[b], 2) for b in NAVS}})
        state["done_days"].append(key)
        del state["open"][key]
    return rows


def load_state():
    st = json.loads(STATE.read_text()) if STATE.exists() else {"nav": {}, "open": {}, "done_days": []}
    for b in NAVS:
        st["nav"].setdefault(b, 100000.0)
    for key, v in list(st["open"].items()):          # Oct 20 format: one SPY entry per date
        if "legs" in v:
            st["open"][key] = {"SPY": {**v, "und": "SPY", "width": 5}}
    return st


def main():
    state = load_state()
    now = dt.datetime.now(ET)
    oc, sc, tc = clients()
    rows = settle(state, oc, sc, now)
    try:
        enter(state, oc, sc, tc, now)
    except Exception as e:
        log.error(f"entry failed: {e}")
        log_event("shadow_condor", LogStatus.ERROR, f"entry failed: {e}")
        notify_alert("Shadow condor", f"entry failed: {e}")
    if rows:
        with LEDGER.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        for r in rows:
            parts = []
            for u, x in r["legs"].items():
                f = lambda p: (f"{x['books'][p]['return_on_risk'] * 100:+.0f}%" if x["books"].get(p) else "n/a")
                parts.append(f"{u} {x['move_pct']:+.2f}% (implied ±{x['implied_move']:.2f}): "
                             f"natural {f('natural')} / mid {f('mid')} / backtest-price {f('vwap1c')}")
            n = r["nav"]
            msg = (f"{r['date']}: " + "; ".join(parts) + f". NAV natural: SPY {n['natural']:,.0f}, QQQ {n['qqq_natural']:,.0f}, "
                   f"combo {n['combo_natural']:,.0f}")
            log.info(msg)
            notify_success("Shadow condor", msg)
            log_event("shadow_condor", LogStatus.SUCCESS, msg, metrics=r["nav"])
    state["done_days"] = state["done_days"][-400:]
    STATE.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
