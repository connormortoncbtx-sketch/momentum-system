"""
research/options_premium_etfs.py -- the SPY premium-selling test on other liquid ETFs (Oct 20, 2026).

Runs research/options_premium.py once per ETF (UNDERLYING env), insurance widths ~0.4% / ~0.8% of the price.
Two readings:
  1. OUT-OF-SAMPLE: the setup chosen on SPY (0DTE iron condor, short strikes 1x the implied move, ~0.8% wings) on
     each ETF, first and second half -- nothing about these ETFs was used to pick it
  2. each ETF's own first-half pick -> its second half (same discipline as SPY)
  3. how correlated the ETFs' daily condor P&L is with SPY's (the only diversification short premium can have)
ETFs without daily expiries simply have no 0DTE trades (1DTE / 7DTE still run).
"""
import os, subprocess, sys
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
ETFS = os.environ.get("ETFS", "QQQ,IWM,DIA,GLD,TLT,SLV,XLE,XLF,EEM").split(",")
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    for u in ETFS:
        print(f"===== {u}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, "options_premium.py")], env={**os.environ, "UNDERLYING": u})
        if r.returncode:
            print(f"::warning::{u}: run failed ({r.returncode})")
    rows, daily = [], {}
    spy = f"{RD}/results/options_premium_trades.parquet"
    for u in ["SPY"] + ETFS:
        p = f"{RD}/results/options_premium_trades{'' if u == 'SPY' else '_' + u}.parquet"
        if not os.path.exists(p):
            continue
        T = pd.read_parquet(p)
        T["W"] = T.W.astype(str)
        if u == "SPY":
            T = T[T.credit < 0.6 * T.W.astype(float)]
        x = T[(T.tenor == "0DTE") & (T.struct == "iron condor") & (T.k == 1.0) & T.W.isin(["5", "0.8%"])]
        for c in (0.01, 0.03):
            for half, m in (("first", x.d < "2025-07-01"), ("second", x.d >= "2025-07-01")):
                y = x[(x.cost == c) & m]
                if len(y) < 30:
                    continue
                dr = y.groupby("d").ror.mean() * 0.02
                rows.append(dict(etf=u, cost=c, half=half, trades=len(y), win_pct=(y.pnl > 0).mean() * 100,
                                 ror_pct=y.ror.mean() * 100, t=y.ror.mean() / y.ror.std() * np.sqrt(len(y)),
                                 sharpe=dr.mean() / dr.std() * np.sqrt(252)))
        y = x[x.cost == 0.01]
        if len(y):
            daily[u] = y.groupby("d").ror.mean()
    R = pd.DataFrame(rows)
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/options_premium_etfs_oos.csv", index=False)
    pd.set_option("display.width", 220)
    print("SPY-CHOSEN SETUP (0DTE iron condor, 1x implied move, ~0.8% wings) ON EACH ETF\n" + R.round(2).to_string(index=False))
    if daily:
        D = pd.DataFrame(daily)
        print("daily P&L correlation with SPY:\n" + D.corr()["SPY"].round(2).to_string() if "SPY" in D else "")
    for u in ETFS:
        s = R[(R.etf == u) & (R.cost == 0.01)].set_index("half")
        if len(s) == 2:
            print(f"::notice::{u} SPY-chosen 0DTE condor @1c: first RoR {s.at['first','ror_pct']:+.1f}% (t {s.at['first','t']:.1f}) "
                  f"-> second {s.at['second','ror_pct']:+.1f}% (t {s.at['second','t']:.1f}), win {s.at['second','win_pct']:.0f}%")
        elif not len(s):
            print(f"::notice::{u}: no 0DTE condor trades (no daily expiries or no data)")


if __name__ == "__main__":
    main()
