"""
research/fetch_events.py
========================
Earnings events from SEC EDGAR + market reaction from Alpaca daily bars.

1. Ticker -> CIK from SEC company_tickers.json (current listings only; delisted
   companies are missing here -- stated with every result).
2. For each CIK, all 8-K filings with Item 2.02 (Results of Operations) since
   START, with the EDGAR acceptance timestamp (Eastern time; EDGAR labels it
   'Z' but it is ET -- verified by the hour histogram this script logs).
3. Daily adjusted bars per symbol -> per-event reaction features:
     pre_date   last session whose close preceded the filing
     d0, d1     first and second sessions after it
     gap        open(d0)/close(pre) - 1
     r2         close(d1)/close(pre) - 1            (2-day reaction)
     r2_abn     r2 - SPY over the same window
     vol60      std of daily returns over 60 sessions before pre_date
     sue        r2_abn / (vol60 * sqrt(2))          (volatility-scaled surprise)
     vol_ratio  volume(d0) / mean volume(20 sessions before pre_date)
   Output: research_data/earnings_events.parquet (merged into research-data branch).

Runs in GitHub Actions (research_events.yml).
"""
import argparse
import datetime as dt
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("fetch_events")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

UA = os.environ.get("SEC_USER_AGENT") or ("MomentumAlpha research momentum-bot@users.noreply.github.com")
START = "2016-01-01"


def sec_get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=30) as r:
                time.sleep(0.12)                     # stay under SEC's 10 req/s
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            log.warning(f"SEC HTTP {e.code} for {url}")
            time.sleep(2 * (i + 1))
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"SEC fetch failed: {url}")


def earnings_filings(symbols: set[str]) -> pd.DataFrame:
    tick = sec_get("https://www.sec.gov/files/company_tickers.json")
    cmap = {}
    for v in tick.values():
        t = v["ticker"].upper().replace("-", ".")
        if t in symbols:
            cmap.setdefault(t, int(v["cik_str"]))
    log.info(f"Mapped {len(cmap):,} of {len(symbols):,} symbols to CIKs")

    rows = []
    seen_cik = {}
    for n, (sym, cik) in enumerate(sorted(cmap.items())):
        if cik in seen_cik:                          # share classes of one issuer
            for r in seen_cik[cik]:
                rows.append({**r, "symbol": sym})
            continue
        sub = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        if not sub:
            continue
        blocks = [sub["filings"]["recent"]]
        for f in sub["filings"].get("files", []):
            if f.get("filingTo", "9999") >= START:
                more = sec_get(f"https://data.sec.gov/submissions/{f['name']}")
                if more:
                    blocks.append(more)
        mine = []
        for b in blocks:
            for form, fdate, acc, items in zip(b["form"], b["filingDate"], b["acceptanceDateTime"], b["items"]):
                if form in ("8-K", "8-K/A") and "2.02" in (items or "") and fdate >= START:
                    mine.append({"symbol": sym, "cik": cik, "form": form, "filing_date": fdate,
                                 "accepted": acc})
        seen_cik[cik] = mine
        rows.extend(mine)
        if n % 500 == 0:
            log.info(f"  EDGAR {n:,}/{len(cmap):,} issuers, {len(rows):,} filings")
    df = pd.DataFrame(rows)
    # EDGAR acceptanceDateTime is Eastern despite the trailing Z
    df["accepted_et"] = pd.to_datetime(df["accepted"].str.replace("Z", "", regex=False).str[:19])
    df = df[df.form == "8-K"].sort_values("accepted_et")
    # one event per symbol per 20 days (8-K amendments / duplicate exhibits)
    df["gap_days"] = df.groupby("symbol")["accepted_et"].diff().dt.days
    df = df[(df.gap_days.isna()) | (df.gap_days > 20)].drop(columns="gap_days")
    hrs = df.accepted_et.dt.hour.value_counts(normalize=True).sort_index().round(3)
    log.info(f"Acceptance-hour distribution (expect peaks ~6-9 and 16-17 if ET): {hrs.to_dict()}")
    return df


def daily_bars(symbols, start, end, feed):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])

    def get(batch):
        for attempt in range(3):
            try:
                req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day, start=start,
                                       end=end, adjustment=Adjustment.ALL, feed=DataFeed(feed))
                df = client.get_stock_bars(req).df
                return df.reset_index() if len(df) else None
            except Exception as e:
                if not any(t in str(e) for t in ("429", "timed out", "502", "503", "504")):
                    break
                time.sleep(5 * (attempt + 1))
        if len(batch) == 1:
            return None
        m = len(batch) // 2
        parts = [x for x in (get(batch[:m]), get(batch[m:])) if x is not None]
        return pd.concat(parts) if parts else None

    for i in range(0, len(symbols), 100):
        df = get(symbols[i:i + 100])
        if df is not None:
            df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
            yield df[["symbol", "date", "open", "close", "volume"]]
        if (i // 100) % 20 == 0:
            log.info(f"  bars {min(i + 100, len(symbols)):,}/{len(symbols):,}")


def reaction_features(ev: pd.DataFrame, bars: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    """Per-event reaction features for one symbol. bars: daily rows for that symbol."""
    b = bars.set_index("date").sort_index()
    if len(b) < 80:
        return pd.DataFrame()
    dates = b.index.values
    close = b["close"].values
    ret = pd.Series(close, index=b.index).pct_change()
    vol60 = ret.rolling(60, min_periods=40).std().values
    avgv20 = b["volume"].rolling(20, min_periods=15).mean().values
    out = []
    for _, e in ev.iterrows():
        t = e.accepted_et
        # last session whose 16:00 close is at or before the filing
        cutoff = t.normalize() if t.hour < 16 else t.normalize() + pd.Timedelta(days=1)
        i_pre = np.searchsorted(dates, np.datetime64(cutoff), side="left") - 1
        if i_pre < 60 or i_pre + 2 >= len(dates):
            continue
        d_pre, d0, d1 = dates[i_pre], dates[i_pre + 1], dates[i_pre + 2]
        if (d0 - d_pre) > np.timedelta64(6, "D"):        # data gap, not a real next session
            continue
        r2 = close[i_pre + 2] / close[i_pre] - 1
        try:
            s2 = spy.loc[d1] / spy.loc[d_pre] - 1
        except KeyError:
            continue
        v = vol60[i_pre]
        out.append({
            "symbol": e.symbol, "accepted_et": t, "pre_date": d_pre, "d0": d0, "d1": d1,
            "gap": b["open"].values[i_pre + 1] / close[i_pre] - 1,
            "r2": r2, "r2_abn": r2 - s2, "vol60": v,
            "sue": (r2 - s2) / (v * np.sqrt(2)) if v and v > 0 else np.nan,
            "vol_ratio": b["volume"].values[i_pre + 1] / avgv20[i_pre] if avgv20[i_pre] else np.nan,
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="research_data")
    ap.add_argument("--feed", default="sip")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    data = Path(a.data)

    assets = pd.read_parquet(data / "assets.parquet")
    syms = set(assets[assets.has_data].symbol)
    if a.limit:
        syms = set(sorted(syms)[: a.limit]) | {"AAPL", "MSFT", "NVDA"}
    ev = earnings_filings(syms)
    ev.to_parquet(data / "earnings_filings_raw.parquet", index=False)
    log.info(f"{len(ev):,} earnings filings for {ev.symbol.nunique():,} symbols")

    end = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    start = dt.datetime(2015, 9, 1, tzinfo=dt.timezone.utc)      # 60-session lookback before 2016
    want = sorted(set(ev.symbol))
    spy_df = next(daily_bars(["SPY"], start, end, a.feed))
    spy = spy_df.set_index("date")["close"]
    feats = []
    by_sym = dict(tuple(ev.groupby("symbol")))
    for chunk in daily_bars(want, start, end, a.feed):
        for sym, g in chunk.groupby("symbol"):
            if sym in by_sym:
                feats.append(reaction_features(by_sym[sym], g, spy))
    out = pd.concat([f for f in feats if len(f)], ignore_index=True)
    out.to_parquet(data / "earnings_events.parquet", index=False)
    log.info(f"Wrote {len(out):,} events with reaction features "
             f"({out.symbol.nunique():,} symbols, {out.d0.min():%Y-%m-%d} -> {out.d0.max():%Y-%m-%d})")
    (data / "README_events.md").write_text(
        f"earnings events built {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC: filings={len(ev)} events={len(out)}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        # Annotations are readable via the API even when job logs aren't.
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
