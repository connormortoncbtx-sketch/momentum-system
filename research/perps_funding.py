"""
research/perps_funding.py -- would a funding-rate carry trade on Kalshi's perpetuals have paid? (Oct 26, 2026)

Kalshi perps (CFTC-regulated, BTC from June 2026, a US large-cap index from Oct 6, 2026) pay a funding rate between
longs and shorts; positive = longs pay shorts. The market-neutral carry trade: SHORT the perp, hold the same notional
of spot (BTC via Alpaca crypto / IBIT; the index via SPY), collect funding while the price moves cancel.
Public endpoint, no keys: GET /margin/funding_rates/historical (all markets). Per market: settlements, spacing,
mean rate per settlement, share positive, annualised simple carry (avg daily x 365, Kalshi's own convention), worst
7-day and 30-day stretches, and by month. Plus the mark price vs spot (BTC) at each funding time = the basis a hedged
position carries. Costs are NOT in the data (fees unknown here): the summary shows break-even round-trip cost for
holding 1 week / 1 month. Aggregates only -> results/perps_funding.csv.
"""
import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
API = "https://external-api.kalshi.com/trade-api/v2"
pd.set_option("display.width", 220)


def get(path, **q):
    url = API + path + ("?" + urllib.parse.urlencode(q) if q else "")
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == 3:
                raise
            time.sleep(2 + 3 * i)


def funding():
    rows, cur, seen = [], None, 0
    for _ in range(200):
        q = {"limit": 1000}
        if cur:
            q["cursor"] = cur
        d = get("/margin/funding_rates/historical", **q)
        fr = d.get("funding_rates") or []
        rows += fr
        cur = d.get("cursor")
        if not cur or not fr or len(rows) == seen:
            break
        seen = len(rows)
    f = pd.DataFrame(rows)
    if f.empty:
        return f
    f["t"] = pd.to_datetime(f.funding_time, utc=True)
    f["rate"] = pd.to_numeric(f.funding_rate, errors="coerce")
    f["mark"] = pd.to_numeric(f.mark_price, errors="coerce")
    return f.drop_duplicates(["market_ticker", "t"]).sort_values(["market_ticker", "t"])


def btc_spot(t0):
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    b = CryptoHistoricalDataClient().get_crypto_bars(CryptoBarsRequest(symbol_or_symbols="BTC/USD", timeframe=TimeFrame.Hour,
                                                                      start=t0)).df.reset_index()
    return b.set_index(pd.to_datetime(b.timestamp, utc=True)).close.sort_index()


def main():
    f = funding()
    if f.empty:
        print("::error::no funding data returned"); return
    print(f"{len(f)} funding settlements, markets: {sorted(f.market_ticker.unique())}")
    print("rate sample (units as returned):", f.rate.describe().round(6).to_dict())
    out = []
    for tk, g in f.groupby("market_ticker"):
        g = g.sort_values("t")
        gap_h = g.t.diff().dt.total_seconds().median() / 3600
        daily = g.set_index("t").rate.groupby(lambda x: x.tz_convert("America/New_York").date()).sum()
        if len(daily) > 2:
            daily = daily.iloc[1:-1]                                  # drop partial first/last day (Kalshi does too)
        roll7, roll30 = daily.rolling(7).sum(), daily.rolling(30).sum()
        out.append(dict(market=tk, settlements=len(g), first=str(g.t.min().date()), last=str(g.t.max().date()),
                        hours_between=round(gap_h, 2), mean_rate=g.rate.mean(), median_rate=g.rate.median(),
                        share_positive=(g.rate > 0).mean(), avg_daily=daily.mean(), annualised=daily.mean() * 365,
                        worst_7d=roll7.min(), best_7d=roll7.max(), worst_30d=roll30.min(), days=len(daily),
                        last_30d_annualised=daily.iloc[-30:].mean() * 365 if len(daily) else np.nan))
        m = daily.groupby(lambda d: f"{d:%Y-%m}").agg(["mean", "count"])
        print(f"\n{tk}: by month, avg daily funding (x365 = annualised)\n" + (m.assign(annualised=m["mean"] * 365)).round(6).to_string())
    S = pd.DataFrame(out)
    print("\nSUMMARY (rates as returned -- if they look like fractions, annualised 0.10 = 10%/yr)\n" + S.round(6).to_string(index=False))
    # basis: BTC mark vs spot at funding times
    b = f[f.market_ticker.str.contains("BTC", case=False)]
    if len(b):
        try:
            sp = btc_spot(b.t.min() - pd.Timedelta("2h"))
            s_at = sp.reindex(b.t.dt.floor("h"), method="ffill").values
            ratio = b.mark.values / s_at
            # the mark may be quoted per a fraction of a coin: express as a ratio to its median
            prem = (ratio / np.nanmedian(ratio) - 1) * 100
            print(f"\nBTC perp mark vs spot (premium vs its own median ratio, %): median {np.nanmedian(prem):+.3f}, "
                  f"p5 {np.nanpercentile(prem, 5):+.3f}, p95 {np.nanpercentile(prem, 95):+.3f}; median mark/spot ratio {np.nanmedian(ratio):.8f}")
        except Exception as e:
            print(f"::warning::spot comparison failed: {str(e)[:150]}")
    os.makedirs(f"{RD}/results", exist_ok=True)
    S.to_csv(f"{RD}/results/perps_funding.csv", index=False)
    print("::notice::perps funding: " + "; ".join(
        f"{r.market}: {r.settlements} settlements every {r.hours_between}h, {r.share_positive:.0%} positive, annualised "
        f"{r.annualised:+.4f} (last 30d {r.last_30d_annualised:+.4f}), worst 30d {r.worst_30d:+.4f}" for r in S.itertuples())
        + " | rates in the endpoint's units (fraction of notional if < 1)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc(); print(tb); print("::error::" + tb[-900:].replace("\n", " | ")); raise
