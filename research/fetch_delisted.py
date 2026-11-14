"""
research/fetch_delisted.py -- fill the 2023-26 delisting hole (Nov 14, 2026).

The weekly panel has only 13-23 stocks a year dropping out after 2022 (several hundred in reality), because it was
built from Alpaca's asset list, which drops delisted names. Alpaca still serves their bars if asked by ticker.
Step 1 (one EODHD request): the list of delisted US tickers (exchange-symbol-list/US?delisted=1). The key comes from
the EODHD_API_KEY secret and is never printed.
Step 2: keep common stocks on NYSE / NASDAQ / NYSE American (and ARCA/BATS listings typed as common stock) whose ticker
is NOT a current Alpaca asset (a reused ticker would splice two companies) and not already in the panel's assets;
fetch daily bars 2016 -> today from Alpaca (SIP, split+dividend adjusted).
Output (research-data branch): delisted/eodhd_us_delisted.csv (the list), daily/delisted_ohlcv_<year>.parquet,
and a summary by year of the last trading day. LIMIT=n: only n tickers (smoke test).
"""
import glob
import io
import json
import os
import time
import urllib.request

import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
LIMIT = int(os.environ.get("LIMIT") or 0)


def eodhd_list():
    key = os.environ.get("EODHD_API_KEY", "").strip()
    if not key:
        raise SystemExit("::error::EODHD_API_KEY secret is missing or empty")
    url = f"https://eodhd.com/api/exchange-symbol-list/US?delisted=1&fmt=json&api_token={key}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "momentum-research"}), timeout=120) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        msg = e.read()[:300].decode("utf-8", "replace").replace(key, "***")
        raise SystemExit(f"::error::EODHD HTTP {e.code}: {msg}")
    data = json.loads(body)
    if not isinstance(data, list):
        raise SystemExit(f"::error::EODHD unexpected reply: {str(data)[:300].replace(key, '***')}")
    return pd.DataFrame(data)


def main():
    L = eodhd_list()
    print(f"EODHD delisted list: {len(L):,} rows; columns {list(L.columns)}", flush=True)
    os.makedirs(f"{RD}/delisted", exist_ok=True)
    L.to_csv(f"{RD}/delisted/eodhd_us_delisted.csv", index=False)
    print("types:", L.get("Type", pd.Series(dtype=str)).value_counts().head(8).to_dict())
    print("exchanges:", L.get("Exchange", pd.Series(dtype=str)).value_counts().head(10).to_dict())
    S = L[(L.get("Type") == "Common Stock")].copy()
    S["Code"] = S.Code.astype(str).str.upper().str.strip()
    S = S[S.Code.str.match(r"^[A-Z]{1,5}$")]
    from alpaca.trading.client import TradingClient
    tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    current = {a.symbol for a in tc.get_all_assets()}
    panel = set(pd.read_parquet(f"{RD}/assets.parquet").symbol)
    have_daily = set()
    for f in glob.glob(f"{RD}/daily/ohlcv_*.parquet"):
        have_daily |= set(pd.read_parquet(f, columns=["symbol"]).symbol.unique())
    cand = sorted(set(S.Code) - current - panel - have_daily)
    print(f"common-stock delisted tickers: {S.Code.nunique():,}; not a current Alpaca asset, not in the panel or daily set: {len(cand):,}", flush=True)
    if LIMIT:
        cand = cand[:LIMIT]
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    c = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    frames, t0 = [], time.time()
    for i in range(0, len(cand), 100):
        b = cand[i:i + 100]
        for attempt in range(4):
            try:
                df = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=b, timeframe=TimeFrame.Day, start=pd.Timestamp("2016-01-01"),
                                                       adjustment=Adjustment.ALL, feed=DataFeed.SIP)).df
                if len(df):
                    df = df.reset_index()[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]
                    df["date"] = pd.to_datetime(df.timestamp).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
                    frames.append(df.drop(columns="timestamp"))
                break
            except Exception as e:
                time.sleep(5 * (attempt + 1))
                if attempt == 3:
                    print(f"::warning::batch {i}: {str(e)[:120]}")
        if (i // 100) % 10 == 0:
            print(f"  {i + 100}/{len(cand)} tickers, {time.time() - t0:.0f}s", flush=True)
    if not frames:
        raise SystemExit("::error::no bars returned")
    D = pd.concat(frames, ignore_index=True)
    for y, g in D.groupby(D.date.dt.year):
        g.sort_values(["symbol", "date"]).to_parquet(f"{RD}/daily/delisted_ohlcv_{y}.parquet", index=False)
    last = D.groupby("symbol").date.max()
    ended = last[last < pd.Timestamp.today().normalize() - pd.Timedelta(days=30)]
    by = ended.groupby(ended.dt.year).size()
    print("tickers with bars whose trading ended, by year:", by.to_dict())
    print(f"::notice::delisted fill: EODHD list {len(L):,}; candidates {len(cand):,}; with Alpaca bars {D.symbol.nunique():,}; "
          f"ended 2023/2024/2025/2026: {by.get(2023, 0)}/{by.get(2024, 0)}/{by.get(2025, 0)}/{by.get(2026, 0)}")


if __name__ == "__main__":
    main()
