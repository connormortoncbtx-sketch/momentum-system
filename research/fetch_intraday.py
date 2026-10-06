"""
research/fetch_intraday.py
==========================
5-minute regular-session bars (Mon 09:30 -> Fri 16:00 ET) for every
(symbol, trade_week) in research/intraday_picks.csv -- the live-momentum
proxy's top 15 each week, 2017-2026. Used to test stop / trail / partial-
profit / entry-timing rules on real intraday paths.

Output: research_data/intraday_5m_<year>.parquet
  trade_week, symbol, ts (ET, naive), o, h, l, c, v
Runs in GitHub Actions (research_intraday.yml).
"""
import datetime as dt
import logging
import os
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger("fetch_intraday")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")


def main():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment, DataFeed

    data = Path(os.environ.get("RESEARCH_DATA", "research_data"))
    picks = pd.read_csv("research/intraday_picks.csv", parse_dates=["trade_week"])
    lim = int(os.environ.get("LIMIT", "0") or 0)
    weeks = sorted(picks.trade_week.unique())
    if lim:
        weeks = weeks[-lim:]
    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    tf = TimeFrame(5, TimeFrameUnit.Minute)
    ny = "America/New_York"

    def get(syms, start, end):
        for attempt in range(3):
            try:
                req = StockBarsRequest(symbol_or_symbols=syms, timeframe=tf, start=start, end=end,
                                       adjustment=Adjustment.ALL, feed=DataFeed.SIP)
                df = client.get_stock_bars(req).df
                return df.reset_index() if len(df) else None
            except Exception as e:
                if not any(t in str(e) for t in ("429", "timed out", "502", "503", "504")):
                    break
                time.sleep(5 * (attempt + 1))
        if len(syms) == 1:
            return None
        m = len(syms) // 2
        parts = [x for x in (get(syms[:m], start, end), get(syms[m:], start, end)) if x is not None]
        return pd.concat(parts) if parts else None

    frames, missing = [], 0
    for n, fri in enumerate(weeks):
        fri = pd.Timestamp(fri)
        syms = sorted(picks[picks.trade_week == fri].symbol.unique())
        mon = fri - pd.Timedelta(days=4)
        start = pd.Timestamp(mon.date()).tz_localize(ny) + pd.Timedelta(hours=9, minutes=30)
        end = pd.Timestamp(fri.date()).tz_localize(ny) + pd.Timedelta(hours=16)
        df = get(syms, start.to_pydatetime(), end.to_pydatetime())
        if df is None:
            missing += len(syms)
            continue
        df["ts"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(ny).dt.tz_localize(None)
        t = df["ts"].dt.time
        df = df[(t >= dt.time(9, 30)) & (t < dt.time(16, 0))]
        missing += len(set(syms) - set(df.symbol))
        out = pd.DataFrame({"trade_week": fri, "symbol": df.symbol, "ts": df.ts,
                            "o": df.open.astype("float32"), "h": df.high.astype("float32"),
                            "l": df.low.astype("float32"), "c": df.close.astype("float32"),
                            "v": df.volume.astype("float32")})
        frames.append(out)
        if n % 50 == 0:
            log.info(f"  {n}/{len(weeks)} weeks, missing symbol-weeks so far {missing}")
    allbars = pd.concat(frames, ignore_index=True)
    for yr, chunk in allbars.groupby(allbars.trade_week.dt.year):
        chunk.to_parquet(data / f"intraday_5m_{yr}.parquet", index=False, compression="zstd")
    msg = (f"intraday 5m: {len(allbars):,} bars, {allbars.groupby(['trade_week','symbol']).ngroups:,} "
           f"symbol-weeks, missing {missing}")
    log.info(msg)
    print(f"::notice::{msg}")
    (data / "README_intraday.md").write_text(msg + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
