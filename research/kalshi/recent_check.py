"""
research/kalshi/recent_check.py -- is the recent losing streak the MARKET or the BOT? (Nov 5, 2026)
Replays the bot's backtest rule (as depth_curve.py: 2c behind, 5-lot, favourite >= 60c, 6-48h, :45-:59 skipped) on the
bot's 11 series for the most recent settled market days, per market day. If the backtest also lost on the days the
bot lost, the streak is market conditions; if the backtest made money while the bot lost, the bot's execution is the
problem. Public data only.
"""
import concurrent.futures as cf, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import depth_curve as DC
DC.N_EV = int(os.environ.get("N_EV") or 14)


def main():
    with cf.ThreadPoolExecutor(4) as ex:
        mk = pd.concat([d for d in ex.map(DC.events, DC.SERIES) if len(d)], ignore_index=True)
    F = []
    with cf.ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(DC.trades, m.ticker): m for m in mk.itertuples()}
        for fu in cf.as_completed(futs):
            try:
                F.append(DC.sim(fu.result(), futs[fu]))
            except Exception as e:
                print(f"::warning::{futs[fu].ticker}: {e}")
    F = pd.concat([f for f in F if len(f)], ignore_index=True)
    F["mday"] = F.event.str.split("-").str[1]
    F["city"] = F.event.str.split("-").str[0].str.replace("KXHIGHT", "").str.replace("KXHIGH", "")
    F["usd"] = F.e * F.c / 100
    d = F.groupby("mday").agg(contracts=("c", "sum"), usd=("usd", "sum"))
    d["edge_c"] = d.usd / d.contracts * 100
    d = d.reindex(sorted(d.index, key=lambda s: pd.to_datetime(s, format="%y%b%d")))
    print("BACKTEST by market day (bot's rule, no cap):\n" + d.round(2).to_string())
    last = d.tail(6)
    print(f"::notice::backtest last {len(last)} market days: " + ", ".join(f"{i} {r.edge_c:+.1f}c" for i, r in last.iterrows())
          + f" | pooled {last.usd.sum() / last.contracts.sum() * 100:+.2f}c vs earlier {d.iloc[:-6].usd.sum() / d.iloc[:-6].contracts.sum() * 100:+.2f}c")
    c = F[F.mday.isin(last.index)].groupby("city").agg(contracts=("c", "sum"), usd=("usd", "sum"))
    c["edge_c"] = c.usd / c.contracts * 100
    print("\nlast days by city:\n" + c.sort_values("edge_c").round(2).to_string())


if __name__ == "__main__":
    main()
