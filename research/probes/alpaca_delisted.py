"""
research/probes/alpaca_delisted.py -- does Alpaca still serve daily bars for stocks delisted 2023-2026 if asked by
ticker, even though they are gone from its asset list? (Nov 13, 2026: the weekly panel has only 13-23 dropouts a year
after 2022 vs several hundred in reality.) If yes, the survivorship hole can be filled for free.

Delisted names = companies with a Form 25 / 25-NSE (exchange delisting) or 15-12B/15-12G (deregistration) filed
2023-2026 in the SEC filing set, with a ticker on file. For each: daily bars 2022-01-01 .. today (SIP, split-adjusted).
Reported: how many return any bars, whether the last bar lands within ~2 weeks of the delisting filing, and how many
are already in the panel. Output: research_data/results/alpaca_delisted_probe.csv. Read-only; no orders.
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fetch_history as FH  # noqa: E402

RD = os.environ.get("RESEARCH_DATA", "research_data")
LIMIT = int(os.environ.get("LIMIT") or 0)


def main():
    cols = ["cik", "tickers", "form", "filing_date"]
    F = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{RD}/sec/filings_202[3-6].parquet"))]
                  + [pd.read_parquet(f"{RD}/sec/filings_delisted.parquet", columns=cols)])
    F = F[F.form.isin(["25", "25-NSE", "15-12B", "15-12G"]) & (F.filing_date >= "2023-01-01")]
    F["t"] = F.tickers.fillna("").str.split(",").str[0].str.strip().str.upper()
    F = F[F.t.str.match(r"^[A-Z]{1,5}$")]
    D = F.sort_values("filing_date").groupby("t").filing_date.first().reset_index()
    panel = set(pd.read_parquet(f"{RD}/assets.parquet").symbol)
    D["in_panel"] = D.t.isin(panel)
    if LIMIT:
        D = D.head(LIMIT)
    print(f"{len(D)} delisted/deregistered tickers 2023-2026 with a ticker on file; {int(D.in_panel.sum())} already in "
          f"the panel's asset list; by year: {D.filing_date.dt.year.value_counts().sort_index().to_dict()}", flush=True)
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    c = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    got = {}
    syms = list(D.t)
    for i in range(0, len(syms), 100):
        b = syms[i:i + 100]
        try:
            df = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=b, timeframe=TimeFrame.Day, start=pd.Timestamp("2022-01-01"),
                                                   adjustment=Adjustment.ALL, feed=DataFeed.SIP)).df
            if len(df):
                df = df.reset_index()
                for s, g in df.groupby("symbol"):
                    got[s] = (g.timestamp.min(), g.timestamp.max(), len(g))
        except Exception as e:
            print(f"::warning::batch {i}: {str(e)[:150]}")
        if i % 1000 == 0:
            print(f"  {i}/{len(syms)}", flush=True)
    D["has_bars"] = D.t.isin(got)
    D["first_bar"] = D.t.map(lambda s: got.get(s, (None,))[0])
    D["last_bar"] = D.t.map(lambda s: got.get(s, (None, None))[1])
    D["n_bars"] = D.t.map(lambda s: got.get(s, (None, None, 0))[2])
    lb = pd.to_datetime(D.last_bar, utc=True).dt.tz_localize(None)
    D["ends_near_delisting"] = (lb - D.filing_date).dt.days.abs() <= 21
    os.makedirs(f"{RD}/results", exist_ok=True)
    D.to_csv(f"{RD}/results/alpaca_delisted_probe.csv", index=False)
    n = len(D); h = int(D.has_bars.sum()); near = int((D.has_bars & D.ends_near_delisting).sum())
    new = int((D.has_bars & ~D.in_panel).sum())
    print(D.groupby([D.filing_date.dt.year, "in_panel"]).agg(n=("t", "size"), has_bars=("has_bars", "sum"),
                                                              ends_near=("ends_near_delisting", "sum")).to_string())
    print(f"::notice::alpaca delisted probe: {n} tickers; {h} return bars ({near} ending within 3 weeks of the delisting "
          f"filing); {new} with bars that the panel does NOT have")


if __name__ == "__main__":
    main()
