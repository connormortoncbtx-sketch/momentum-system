"""
research/fetch_daily.py -- daily OHLCV for every stock that was ever tradeable in the research universe (price >= $5
and >= $2M/day in some week of the weekly panel), 2016 -> today, split- and dividend-adjusted (Alpaca, SIP feed).
Needed by chart_cnn.py (the price-chart image model, Jiang, Kelly & Xiu 2023), Nov 13, 2026.
Output: research_data/daily/ohlcv_<year>.parquet  (symbol, date, open, high, low, close, volume)
Runs in Actions (research_run.yml) where the Alpaca keys live. LIMIT=n fetches only n symbols (smoke test).
"""
import glob
import os
import time

import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = f"{RD}/daily"
LIMIT = int(os.environ.get("LIMIT") or 0)
BATCH = 100


def universe():
    L = pd.concat([pd.read_parquet(f, columns=["symbol", "raw_close", "dollar_vol", "n_days"])
                   for f in glob.glob(f"{RD}/weekly_*.parquet")])
    L = L[(L.raw_close >= 5) & (L.dollar_vol / L.n_days >= 2e6)]
    syms = sorted(set(L.symbol))
    funds = set()
    try:
        a = pd.read_parquet(f"{RD}/assets.parquet")
        if "is_fund" in a:
            funds = set(a[a.is_fund].symbol)
    except Exception:
        pass
    return [s for s in syms if s not in funds]


def main():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    c = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    syms = universe()
    if LIMIT:
        syms = syms[:LIMIT]
    print(f"{len(syms)} symbols", flush=True)
    frames, failed = [], []

    def get(b, depth=0):
        for attempt in range(4):
            try:
                df = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=b, timeframe=TimeFrame.Day,
                                                       start=pd.Timestamp("2016-01-01"), adjustment=Adjustment.ALL,
                                                       feed=DataFeed.SIP)).df
                if len(df):
                    df = df.reset_index()[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]
                    df["date"] = pd.to_datetime(df.timestamp).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
                    frames.append(df.drop(columns="timestamp"))
                return
            except Exception as e:
                msg = str(e)
                if any(t in msg for t in ("429", "timed out", "Timeout", "502", "503", "504")):
                    time.sleep(5 * (attempt + 1)); continue
                break
        if len(b) == 1:
            failed.extend(b); return
        get(b[:len(b) // 2], depth + 1); get(b[len(b) // 2:], depth + 1)

    t0 = time.time()
    for i in range(0, len(syms), BATCH):
        get(syms[i:i + BATCH])
        if (i // BATCH) % 5 == 0:
            print(f"  {i + BATCH}/{len(syms)} symbols, {time.time() - t0:.0f}s", flush=True)
    D = pd.concat(frames, ignore_index=True)
    for col in ("open", "high", "low", "close"):
        D[col] = D[col].astype("float32")
    D["volume"] = D["volume"].astype("float64")
    os.makedirs(OUT, exist_ok=True)
    for y, g in D.groupby(D.date.dt.year):
        g.sort_values(["symbol", "date"]).to_parquet(f"{OUT}/ohlcv_{y}.parquet", index=False)
    print(f"::notice::daily bars: {D.symbol.nunique()} symbols, {len(D):,} rows, {D.date.min().date()}..{D.date.max().date()}; "
          f"{len(failed)} failed")


if __name__ == "__main__":
    main()
