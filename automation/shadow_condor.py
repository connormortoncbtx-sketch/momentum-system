"""
automation/shadow_condor.py
===========================
Shadow #9 (no orders): same-day SPY iron condor (research/options_premium.py, Oct 20, 2026).

  setup     each trading day at ~10:00 ET, an iron condor expiring today on SPY: short call and short put one
            implied move (ATM straddle mid) from the price, insurance $5 further out; held to the 4pm close
  sizing    2% of the book's NAV at risk (the $5 width minus the credit) per day
  books     NATURAL  sell at the bid, buy at the ask (worst case, live quotes at entry)
            MID      quote midpoints (best case)
            VWAP1C   the backtest's price: 10:00-10:15 trade VWAP per leg, $0.01 per leg cost (filled at settlement)
  backtest  0DTE condor, 1x implied move, $5 wings, $0.01/leg: RoR +3.1%/trade both halves (Sharpe 1.55 -> 1.69),
            win 74%; at $0.03/leg +1.2%/trade (t 1.0). The cost of a real fill decides it -- that is what the
            NATURAL vs MID vs VWAP1C books measure.

It places no orders (Alpaca paper fills are optimistic, so paper orders would not answer the cost question).
Quotes come from the OPRA feed when the account has it, else Alpaca's indicative feed (logged per entry).
Each run: settle earlier days' entries (next morning, so same-day option bars are available), then enter today if inside 09:58-11:30 ET and not yet entered.
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
UND, K, WIDTH, RISK_FRAC = "SPY", 1.0, 5, 0.02
ENTRY_FROM, ENTRY_TO = dt.time(9, 58), dt.time(11, 30)
BOOKS = ("natural", "mid", "vwap1c")


def occ(exp, cp, k):
    return f"{UND}{exp:%y%m%d}{cp}{int(round(k * 1000)):08d}"


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


def spot_now(sc):
    from alpaca.data.requests import StockLatestTradeRequest
    from alpaca.data.enums import DataFeed
    for feed in (DataFeed.SIP, DataFeed.IEX):
        try:
            t = sc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=UND, feed=feed))[UND]
            return float(t.price), feed.value
        except Exception as e:
            log.info(f"spot via {feed.value}: {e}")
    raise RuntimeError("no SPY price")


def official_close(sc, day):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    end = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20)
    df = sc.get_stock_bars(StockBarsRequest(symbol_or_symbols=UND, timeframe=TimeFrame.Day,
                                            start=dt.datetime.combine(day, dt.time()), end=end,
                                            adjustment=Adjustment.RAW, feed=DataFeed.SIP)).df.reset_index()
    df["date"] = df.timestamp.dt.tz_convert(ET).dt.date
    r = df[df.date == day]
    return float(r.close.iloc[0]) if len(r) else None


def vwap_prices(oc, day, syms):
    """10:00-10:15 ET trade VWAP per leg -- the backtest's entry price. Untraded wing -> $0.01."""
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


def enter(state, oc, sc, tc, now):
    day = now.date()
    if day.isoformat() in state["open"] or day.isoformat() in state["done_days"]:
        return
    if not (ENTRY_FROM <= now.time() <= ENTRY_TO):
        if now.time() > ENTRY_TO and now.weekday() < 5:
            log.info("entry window passed without an entry today")
        return
    clock = tc.get_clock()
    if not clock.is_open:
        log.info("market closed today -- no entry"); return
    s0, sfeed = spot_now(sc)
    atm = round(s0)
    q1, feed = quotes(oc, [occ(day, "C", atm), occ(day, "P", atm)])
    mid = lambda q: (q["bid"] + q["ask"]) / 2
    M = mid(q1[occ(day, "C", atm)]) + mid(q1[occ(day, "P", atm)])
    ps, cs = round(s0 - K * M), round(s0 + K * M)
    legs = {"short_put": occ(day, "P", ps), "long_put": occ(day, "P", ps - WIDTH),
            "short_call": occ(day, "C", cs), "long_call": occ(day, "C", cs + WIDTH)}
    q, feed2 = quotes(oc, list(legs.values()))
    Q = {k: q[v] for k, v in legs.items()}
    natural = Q["short_put"]["bid"] + Q["short_call"]["bid"] - Q["long_put"]["ask"] - Q["long_call"]["ask"]
    midc = sum((1 if k.startswith("short") else -1) * mid(Q[k]) for k in Q)
    e = {"date": day.isoformat(), "entered_at": now.isoformat(timespec="seconds"), "spot": s0, "spot_feed": sfeed,
         "quote_feed": feed2, "implied_move": round(M, 4), "put_strikes": [ps, ps - WIDTH], "call_strikes": [cs, cs + WIDTH],
         "legs": legs, "quotes": Q, "credit": {"natural": round(natural, 4), "mid": round(midc, 4)},
         "leg_spread_avg": round(sum(Q[k]["ask"] - Q[k]["bid"] for k in Q) / 4, 4)}
    state["open"][day.isoformat()] = e
    log.info(f"entered {day}: SPY {s0:.2f}, move {M:.2f}, puts {ps}/{ps - WIDTH}, calls {cs}/{cs + WIDTH}, "
             f"credit natural {natural:.2f} / mid {midc:.2f} ({feed2})")


def settle(state, oc, sc, now):
    rows = []
    for key in sorted(state["open"]):
        e = state["open"][key]
        day = dt.date.fromisoformat(key)
        if day >= now.date():          # settle the next trading morning: without a signed OPRA agreement Alpaca refuses
            continue                   # same-day option bars, which the backtest-price (VWAP1C) book needs
        sT = official_close(sc, day)
        if sT is None:
            log.info(f"{key}: close not available yet"); continue
        ps, cs = e["put_strikes"][0], e["call_strikes"][0]
        loss = min(max(ps - sT, 0), WIDTH) + min(max(sT - cs, 0), WIDTH)
        credits = dict(e["credit"])
        try:
            v = vwap_prices(oc, day, list(e["legs"].values()))
            px = {k: v.get(s, 0.01 if k.startswith("long") else None) for k, s in e["legs"].items()}
            if None not in px.values():
                credits["vwap1c"] = round(px["short_put"] + px["short_call"] - px["long_put"] - px["long_call"] - 4 * 0.01, 4)
        except Exception as ex:
            log.warning(f"{key}: vwap prices: {ex}")
        res = {}
        for b in BOOKS:
            c = credits.get(b)
            if c is None or c <= 0 or c >= WIDTH:
                res[b] = None; continue                      # no valid trade in this book (e.g. natural credit <= 0)
            ror = (c - loss) / (WIDTH - c)
            state["nav"][b] *= 1 + RISK_FRAC * ror
            res[b] = {"credit": c, "pnl_per_share": round(c - loss, 4), "return_on_risk": round(ror, 4)}
        row = {"date": key, "spot_entry": e["spot"], "spy_close": sT, "move_pct": round((sT / e["spot"] - 1) * 100, 3),
               "implied_move": e["implied_move"], "put_strikes": e["put_strikes"], "call_strikes": e["call_strikes"],
               "loss_per_share": round(loss, 4), "quote_feed": e["quote_feed"], "leg_spread_avg": e["leg_spread_avg"],
               "entered_at": e["entered_at"], "books": res, "nav": {b: round(state["nav"][b], 2) for b in BOOKS}}
        rows.append(row)
        state["done_days"].append(key)
        del state["open"][key]
    return rows


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "nav": {b: 100000.0 for b in BOOKS}, "open": {}, "done_days": []}
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
        r = rows[-1]
        fmt = lambda b: (f"{r['books'][b]['return_on_risk'] * 100:+.0f}%" if r["books"].get(b) else "n/a")
        msg = (f"{r['date']}: SPY {r['move_pct']:+.2f}% vs implied ±{r['implied_move']:.2f}; "
               f"return on risk natural {fmt('natural')} / mid {fmt('mid')} / backtest-price {fmt('vwap1c')}. "
               f"NAV " + ", ".join(f"{b} {v:,.0f}" for b, v in r["nav"].items()))
        log.info(msg)
        notify_success("Shadow condor", msg)
        log_event("shadow_condor", LogStatus.SUCCESS, msg, metrics=r["nav"])
    state["done_days"] = state["done_days"][-400:]
    STATE.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
