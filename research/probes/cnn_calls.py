"""
research/probes/cnn_calls.py -- could the CNN's bottom decile be played by WRITING CALLS (bear call spreads) instead of
shorting? (Nov 14, 2026.) First gate: do these stocks even have usable options?

For every stock in the CNN's bottom 10% over the latest 8 weeks of predictions:
  - listed call contracts expiring 14-60 days out (Alpaca option contracts API)
  - for those that have them: the call nearest 10% out of the money at the expiry nearest 30 days -- open interest,
    bid / ask from the free indicative snapshot feed, spread as % of mid, and the bid as % of the stock price
Compared with the same measures for the CNN's middle (40-60th percentile) names as a reference.
Read-only; no orders. Output: results/cnn_calls.csv.
"""
import glob
import os
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
pd.set_option("display.width", 200)


def main():
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import ContractType
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionSnapshotRequest
    from alpaca.data.enums import OptionsFeed
    key, sec = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    tc, oc = TradingClient(key, sec, paper=True), OptionHistoricalDataClient(key, sec)
    P = pd.read_parquet(f"{RD}/results/chart_cnn_preds.parquet")
    P["pct"] = P.groupby("week").cnn.rank(pct=True)
    last = P.week.max()
    L = P[P.week >= last - pd.Timedelta(weeks=7)]
    groups = {"CNN bottom 10%": sorted(set(L[L.pct <= 0.10].symbol)),
              "CNN middle (40-60%)": sorted(set(L[(L.pct >= 0.4) & (L.pct <= 0.6)].symbol))}
    D = pd.read_parquet(sorted(glob.glob(f"{RD}/daily/ohlcv_*.parquet"))[-1], columns=["symbol", "date", "close"])
    px = D.sort_values("date").groupby("symbol").close.last()
    rows = []
    today = date.today()
    for g, syms in groups.items():
        rng = np.random.default_rng(0)
        if len(syms) > 400:
            syms = sorted(rng.choice(syms, 400, replace=False))
        contracts = []
        for i in range(0, len(syms), 50):
            b = syms[i:i + 50]
            token = None
            while True:
                try:
                    req = GetOptionContractsRequest(underlying_symbols=b, type=ContractType.CALL,
                                                    expiration_date_gte=today + timedelta(days=14),
                                                    expiration_date_lte=today + timedelta(days=60), limit=10000, page_token=token)
                    res = tc.get_option_contracts(req)
                except Exception as e:
                    print(f"::warning::contracts {b[:3]}: {str(e)[:120]}"); break
                for c in res.option_contracts or []:
                    contracts.append(dict(underlying=c.underlying_symbol, symbol=c.symbol, exp=pd.Timestamp(c.expiration_date),
                                          strike=float(c.strike_price), oi=float(c.open_interest or 0)))
                token = res.next_page_token
                if not token:
                    break
        Cc = pd.DataFrame(contracts)
        have = set(Cc.underlying) if len(Cc) else set()
        pick = []
        for u, x in (Cc.groupby("underlying") if len(Cc) else []):
            s = px.get(u, np.nan)
            if not np.isfinite(s):
                continue
            x = x.assign(dte=(x.exp - pd.Timestamp(today)).dt.days)
            e = x.iloc[(x.dte - 30).abs().argsort()].exp.iloc[0]
            y = x[x.exp == e]
            c = y.iloc[(y.strike - 1.10 * s).abs().argsort()].iloc[0]
            pick.append(dict(group=g, underlying=u, stock=s, option=c.symbol, strike=c.strike, dte=int((e - pd.Timestamp(today)).days), oi=c.oi))
        Pk = pd.DataFrame(pick)
        snaps = {}
        for i in range(0, len(Pk), 100):
            b = list(Pk.option.iloc[i:i + 100])
            try:
                sn = oc.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=b, feed=OptionsFeed.INDICATIVE))
                for k, v in sn.items():
                    q = v.latest_quote
                    snaps[k] = (float(q.bid_price) if q else np.nan, float(q.ask_price) if q else np.nan)
            except Exception as e:
                print(f"::warning::snapshots: {str(e)[:150]}")
            time.sleep(0.3)
        if len(Pk):
            Pk["bid"] = Pk.option.map(lambda k: snaps.get(k, (np.nan, np.nan))[0])
            Pk["ask"] = Pk.option.map(lambda k: snaps.get(k, (np.nan, np.nan))[1])
            Pk["mid"] = (Pk.bid + Pk.ask) / 2
            Pk["spread_pct_mid"] = (Pk.ask - Pk.bid) / Pk.mid * 100
            Pk["bid_pct_stock"] = Pk.bid / Pk.stock * 100
            rows.append(Pk)
        n = len(syms)
        print(f"\n{g}: {n} names checked; with calls expiring in 14-60 days: {len(have)} ({len(have) / n:.0%})")
        if len(Pk):
            q = Pk[Pk.bid > 0]
            print(f"  ~10% OTM call near 30 days: open interest median {Pk.oi.median():.0f} (zero OI: {(Pk.oi == 0).mean():.0%}); "
                  f"a live bid: {len(q)} of {len(Pk)}; bid-ask spread median {q.spread_pct_mid.median():.0f}% of mid "
                  f"(75th pct {q.spread_pct_mid.quantile(.75):.0f}%); bid median {q.bid_pct_stock.median():.1f}% of the stock price")
    R = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/cnn_calls.csv", index=False)
    if len(R):
        b = R[R.group == "CNN bottom 10%"]
        print(f"::notice::CNN calls: bottom-10% names with a live bid on a ~10% OTM ~30-day call: {int((b.bid > 0).sum())}; "
              f"median spread {b[b.bid > 0].spread_pct_mid.median():.0f}% of mid")


if __name__ == "__main__":
    main()
