"""
research/options_straddles.py -- do our "big move coming" predictions beat option prices?

For each (week, symbol) in research/samples/straddle_targets.parquet (model picks and
random controls, Feb 2024 - Sep 2026, all out-of-sample for the model):
  * Friday close S0; the nearest weekly expiry 5-12 days out; the strike closest to S0
    with both a call and a put listed (Alpaca contract list, expired contracts included)
  * entry premium = that Friday's VWAP of the call + the put (daily option bars;
    Alpaca option history starts early 2024); missing if either leg did not trade
  * payoff at expiry = |S_T - K| from the stock's close on the expiry date
  * straddle return = payoff / premium - 1; implied move = premium / S0
Output: research_data/options/straddles.parquet. Runs in GitHub Actions (research_run.yml).
"""
import datetime as dt
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("straddles")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
OUT = Path(os.environ.get("RESEARCH_DATA", "research_data")) / "options"
K, S = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]


class Limiter:
    def __init__(self, per_sec):
        self.gap, self.lock, self.t = 1.0 / per_sec, threading.Lock(), 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.t:
                time.sleep(self.t - now)
            self.t = max(now, self.t) + self.gap


LIM = Limiter(3)


def call(fn, *a, tries=4, **k):
    for i in range(tries):
        LIM.wait()
        try:
            return fn(*a, **k)
        except Exception as e:
            if i == tries - 1:
                log.warning(f"  {fn.__name__} failed: {type(e).__name__}: {str(e)[:120]}")
                return None
            time.sleep(3 * (i + 1))


def main():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import AssetStatus

    OUT.mkdir(parents=True, exist_ok=True)
    tg = pd.read_parquet("research/samples/straddle_targets.parquet")
    lim = int(os.environ.get("LIMIT") or 0)
    if lim:
        tg = tg.groupby("group").head(lim)
    sc = StockHistoricalDataClient(K, S); oc = OptionHistoricalDataClient(K, S); tc = TradingClient(K, S, paper=True)

    # stock closes (RAW prices: option strikes are in raw terms)
    syms = sorted(tg.symbol.unique())
    frames = []
    for i in range(0, len(syms), 200):
        r = call(sc.get_stock_bars, StockBarsRequest(symbol_or_symbols=syms[i:i + 200], timeframe=TimeFrame.Day,
                 start=dt.datetime(2024, 1, 15), adjustment=Adjustment.RAW, feed=DataFeed.SIP))
        if r is not None and len(r.df):
            frames.append(r.df.reset_index()[["symbol", "timestamp", "close"]])
    sb = pd.concat(frames)
    sb["date"] = pd.to_datetime(sb.timestamp).dt.tz_convert("America/New_York").dt.date
    close = sb.pivot_table(index="date", columns="symbol", values="close")
    log.info(f"stock closes: {close.shape}")

    rows = []
    today = dt.date.today()
    for n, r in enumerate(tg.itertuples(index=False)):
        w = pd.Timestamp(r.week).date()
        rec = {"week": r.week, "symbol": r.symbol, "group": r.group, "p_move": r.p_move, "status": "ok"}
        if r.symbol not in close.columns or w not in close.index or not np.isfinite(close.at[w, r.symbol]):
            rows.append({**rec, "status": "no stock close"}); continue
        s0 = float(close.at[w, r.symbol])
        cons = []
        for status in (AssetStatus.INACTIVE, AssetStatus.ACTIVE):
            if status == AssetStatus.ACTIVE and w + dt.timedelta(days=12) < today - dt.timedelta(days=3):
                continue
            resp = call(tc.get_option_contracts, GetOptionContractsRequest(
                underlying_symbols=[r.symbol], status=status,
                expiration_date_gte=w + dt.timedelta(days=5), expiration_date_lte=w + dt.timedelta(days=12),
                strike_price_gte=str(round(s0 * 0.7, 2)), strike_price_lte=str(round(s0 * 1.3, 2)), limit=500))
            if resp is not None and resp.option_contracts:
                cons += resp.option_contracts
        if not cons:
            rows.append({**rec, "status": "no options"}); continue
        cdf = pd.DataFrame([{"sym": c.symbol, "exp": c.expiration_date, "k": float(c.strike_price),
                             "type": str(c.type).lower()} for c in cons]).drop_duplicates("sym")
        exp = cdf.exp.min()
        cdf = cdf[cdf.exp == exp]
        pairs = cdf.pivot_table(index="k", columns="type", values="sym", aggfunc="first").dropna()
        pairs.columns = [c.split(".")[-1] for c in pairs.columns]
        if pairs.empty or not {"call", "put"} <= set(pairs.columns):
            rows.append({**rec, "status": "no call+put pair"}); continue
        kk = pairs.index[np.argmin(np.abs(pairs.index.values - s0))]
        cs, ps = pairs.loc[kk, "call"], pairs.loc[kk, "put"]
        ob = call(oc.get_option_bars, OptionBarsRequest(symbol_or_symbols=[cs, ps], timeframe=TimeFrame.Day,
                  start=dt.datetime.combine(w, dt.time()), end=dt.datetime.combine(w + dt.timedelta(days=1), dt.time())))
        d = ob.df.reset_index() if ob is not None and len(ob.df) else pd.DataFrame()
        px = {}
        for s_ in (cs, ps):
            x = d[d.symbol == s_] if len(d) else d
            if len(x):
                v = x.iloc[-1]
                px[s_] = (float(v.vwap) if np.isfinite(v.vwap) and v.vwap > 0 else float(v.close), float(v.volume))
        if len(px) < 2:
            rows.append({**rec, "status": "leg not traded", "strike": kk, "expiry": str(exp)}); continue
        prem = px[cs][0] + px[ps][0]
        expd = exp if isinstance(exp, dt.date) else pd.Timestamp(exp).date()
        last = [x for x in close.index if w < x <= expd]
        if not last or not np.isfinite(close.at[last[-1], r.symbol]):
            rows.append({**rec, "status": "no expiry close"}); continue
        st = float(close.at[last[-1], r.symbol])
        rows.append({**rec, "s0": s0, "strike": kk, "expiry": str(expd), "premium": prem,
                     "call_vol": px[cs][1], "put_vol": px[ps][1], "s_t": st,
                     "implied_move": prem / s0, "realized_move": abs(st - s0) / s0,
                     "payoff": abs(st - kk), "ret": abs(st - kk) / prem - 1})
        if n % 250 == 0:
            log.info(f"  {n:,}/{len(tg):,} done; ok so far {sum(1 for x in rows if x['status'] == 'ok'):,}")
    res = pd.DataFrame(rows)
    res.to_parquet(OUT / "straddles.parquet", index=False)
    ok = res[res.status == "ok"]
    summ = ok.groupby("group").agg(n=("ret", "size"), mean_ret=("ret", "mean"), median_ret=("ret", "median"),
                                   implied=("implied_move", "median"), realized=("realized_move", "median"))
    print("::notice::straddles " + " | ".join(f"{g}: n {int(r.n)}, mean {r.mean_ret:+.3f}, median {r.median_ret:+.3f}, "
                                               f"implied {r.implied:.3f}, realized {r.realized:.3f}" for g, r in summ.iterrows())
          + f" | status counts {res.status.value_counts().to_dict()}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
