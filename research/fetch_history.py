"""
research/fetch_history.py
=========================
Pull ~10 years of split/dividend-adjusted daily bars for every US equity
Alpaca knows about (active AND inactive, to limit survivorship bias) and
condense them into a compact weekly panel for backtesting.

Runs in GitHub Actions (research_data.yml) because that's where the Alpaca
keys live. Output: research_data/weekly_<year>.parquet + assets.parquet,
force-pushed to the orphan `research-data` branch (kept off main).

Weekly panel columns (one row per symbol per trading week, keyed by the
Friday date of that calendar week):
    first_date     first trading day of the week
    n_days         trading days in the week
    d1_open        first day's open          (Monday open, normally)
    d1_close       first day's close         (Monday close -> designed entry)
    d2_open        second day's open         (Tuesday open)
    close          last day's close          (Friday close -> designed exit)
    low_after_d1   min low over days 2..n    (for stop simulation after Mon-close entry)
    low, high      weekly extremes
    dollar_vol     sum(close * volume)       (liquidity filter)
"""
import argparse
import datetime as dt
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("fetch_history")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")

EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}
BATCH = 200


def list_symbols():
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus

    tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    rows = []
    for status in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
        assets = tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=status))
        for a in assets:
            ex = str(getattr(a.exchange, "value", a.exchange))
            if ex not in EXCHANGES:
                continue
            rows.append({"symbol": a.symbol, "name": a.name or "", "exchange": ex,
                         "status": str(getattr(a.status, "value", a.status))})
    df = pd.DataFrame(rows).drop_duplicates("symbol")
    # Drop obvious non-common share classes that clutter the panel
    df = df[~df.symbol.str.contains(r"[./]", regex=True)]
    log.info(f"Assets: {len(df):,} ({(df.status == 'active').sum():,} active, "
             f"{(df.status != 'active').sum():,} inactive)")
    return df


def fetch_daily(symbols, start, end, feed):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed

    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    frames, failed = [], []
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i + BATCH]
        for attempt in range(4):
            try:
                req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                                       start=start, end=end, adjustment=Adjustment.ALL,
                                       feed=DataFeed(feed))
                df = client.get_stock_bars(req).df
                if len(df):
                    df = df.reset_index()[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]
                    frames.append(weekly(df))
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                log.warning(f"  batch {i // BATCH} attempt {attempt + 1} failed: {e} -- retry in {wait}s")
                time.sleep(wait)
        else:
            failed.extend(batch)
        if (i // BATCH) % 10 == 0:
            log.info(f"  {min(i + BATCH, len(symbols)):,}/{len(symbols):,} symbols")
    if failed:
        log.warning(f"{len(failed)} symbols failed after retries")
    return pd.concat(frames, ignore_index=True), failed


def weekly(d: pd.DataFrame) -> pd.DataFrame:
    """Condense one batch of daily bars into the weekly panel."""
    d = d.copy()
    d["date"] = pd.to_datetime(d["timestamp"]).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    d["week"] = d["date"] + pd.to_timedelta(4 - d["date"].dt.weekday, unit="D")  # Friday of that week
    d = d.sort_values(["symbol", "date"])
    d["k"] = d.groupby(["symbol", "week"]).cumcount()
    d["dv"] = d["close"] * d["volume"]
    g = d.groupby(["symbol", "week"])
    out = g.agg(first_date=("date", "first"), n_days=("date", "size"),
                d1_open=("open", "first"), d1_close=("close", "first"),
                close=("close", "last"), low=("low", "min"), high=("high", "max"),
                dollar_vol=("dv", "sum"))
    out["d2_open"] = d[d.k == 1].set_index(["symbol", "week"])["open"]
    out["low_after_d1"] = d[d.k >= 1].groupby(["symbol", "week"])["low"].min()
    out = out.reset_index()
    for c in ["d1_open", "d1_close", "d2_open", "close", "low", "high", "low_after_d1"]:
        out[c] = out[c].astype("float32")
    out["dollar_vol"] = out["dollar_vol"].astype("float64")
    out["n_days"] = out["n_days"].astype("int8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--feed", default="sip")
    ap.add_argument("--out", default="research_data")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N symbols")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assets = list_symbols()
    syms = assets.symbol.tolist()
    for must in ("SPY", "IWM", "QQQ"):
        if must not in syms:
            syms.insert(0, must)
    if a.limit:
        syms = syms[: a.limit]

    end = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    start = dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc)
    t0 = time.time()
    panel, failed = fetch_daily(syms, start, end, a.feed)
    log.info(f"Fetched {panel.symbol.nunique():,} symbols, {len(panel):,} symbol-weeks "
             f"in {(time.time() - t0) / 60:.1f} min")

    assets["has_data"] = assets.symbol.isin(set(panel.symbol))
    assets.to_parquet(out / "assets.parquet", index=False)
    for yr, chunk in panel.groupby(panel.week.dt.year):
        chunk.to_parquet(out / f"weekly_{yr}.parquet", index=False, compression="zstd")
        log.info(f"  weekly_{yr}.parquet: {len(chunk):,} rows, "
                 f"{(out / f'weekly_{yr}.parquet').stat().st_size / 1e6:.1f} MB")
    (out / "README.md").write_text(
        f"Generated {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC by research/fetch_history.py\n"
        f"start={a.start} feed={a.feed} symbols_requested={len(syms)} "
        f"symbols_with_data={panel.symbol.nunique()} failed={len(failed)}\n")


if __name__ == "__main__":
    main()
