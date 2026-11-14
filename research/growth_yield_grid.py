"""
research/growth_yield_grid.py -- the growth + yield index over a full grid (Nov 13, 2026; Connor: "experiment with
different X number of stocks, and rolling window sizes of conditions being met").

Same measures as growth_yield_index.py (beat = price growth > SPY's AND dividend yield > SPY's over a W-week window).
Grid:
  X        5, 10, 15, 20, 30, 50, 75, 100 stocks
  W        13, 26, 52, 104, 156 weeks (the window over which a stock must beat SPY on both)
  SPAN     0, 26, 52, 104 weeks of window END dates looked back over (every 13 weeks): the score is the share of those
           windows in which the stock beat on both ("continuously"), ties broken by average excess total return;
           SPAN 0 = only the latest window, ranked by the smaller of the two margins. W + SPAN <= 156.
  REBAL    every 4 weeks or every 13 weeks; a holding is kept while it ranks within 2X
Selection discipline (256 variants -> some will beat SPY by luck): pick the best variants on 2019-2022 ONLY, then read
them on 2023-Sep 2026. Also reported: the share of all variants that beat SPY in each period.
Universe price >= $5, ADV >= $5M, 3+ years listed; spread costs; holdings from 2019. Public data only.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as B  # noqa: E402
import universe_floor as U  # noqa: E402

RD = os.environ.get("RESEARCH_DATA", "research_data")
SPLIT = pd.Timestamp("2023-01-01")
START = pd.Timestamp("2019-01-04")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)


def main():
    p = B.load_panel(RD); f = B.features(p)
    C, Rw = p.df("close"), p.df("raw_close")
    dfac = np.log(C / Rw).diff().where(lambda x: x.abs() <= 0.15, 0.0).clip(lower=0)
    tr = np.log(C / C.shift(1)).where(lambda x: x.abs() < 1.5)
    base = (C.notna().cumsum() >= 156) & ~pd.DataFrame(np.tile(p.is_fund, (len(p.weeks), 1)), index=p.weeks, columns=p.symbols)
    m = base & (f["price"] >= 5) & (f["adv"] >= 5e6)
    small = (f["price"] < 5) | (f["adv"] < 2e6)
    cost = pd.DataFrame(U.half_spread_bps(f["price"].fillna(0).to_numpy(), f["adv"].fillna(0).to_numpy()), index=p.weeks, columns=p.symbols)
    d1c = p.df("d1_close"); wr = (d1c.shift(-2) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 1.5)
    spy = wr["SPY"]
    idx = p.weeks[p.weeks >= START]
    mS, wrS, cS, sS = m.loc[idx], wr.loc[idx], cost.loc[idx], small.loc[idx]
    rows = []
    for W in (13, 26, 52, 104, 156):
        dy = dfac.rolling(W, min_periods=int(W * 0.9)).sum()
        pg = tr.rolling(W, min_periods=int(W * 0.9)).sum() - dy
        mpg, mdy = pg.sub(pg["SPY"], axis=0), dy.sub(dy["SPY"], axis=0)
        beat = ((mpg > 0) & (mdy > 0)).astype(float)
        exc = (mpg + mdy)
        for SPAN in (0, 26, 52, 104):
            if W + SPAN > 156:
                continue
            if SPAN == 0:
                z = np.minimum(mpg.div(mpg.std(axis=1), axis=0), mdy.div(mdy.std(axis=1), axis=0))
                score = z.where(beat > 0)
            else:
                lags = range(0, SPAN + 1, 13)
                cons = sum(beat.shift(l) for l in lags) / len(lags)
                ae = sum(exc.shift(l) for l in lags) / len(lags)
                score = (cons + 0.001 * ae.rank(axis=1, pct=True)).where(cons > 0)
            score = score.where(m).loc[idx]
            for REB in (4, 13):
                for X in (5, 10, 15, 20, 30, 50, 75, 100):
                    r = U.sim(score, mS, wrS, cS, sS, n=X, k=REB, buffer=2 * X)
                    row = dict(W=W, SPAN=SPAN, REBAL=REB, X=X)
                    for lab, rr in (("dev", r[r.index < SPLIT]), ("hold", r[r.index >= SPLIT]), ("all", r)):
                        st = U.st(rr); sp = U.st(spy.reindex(rr.index))
                        row.update({f"{lab}_CAGR": st["CAGR"], f"{lab}_Sharpe": st["Sharpe"], f"{lab}_DD": st["maxDD"],
                                    f"{lab}_vsSPY": st["CAGR"] - sp["CAGR"], f"{lab}_SPYSharpe": sp["Sharpe"]})
                    rows.append(row)
            print(f"  W={W} SPAN={SPAN} done", flush=True)
    R = pd.DataFrame(rows)
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/growth_yield_grid.csv", index=False)
    sp_d, sp_h = R.dev_SPYSharpe.iloc[0], R.hold_SPYSharpe.iloc[0]
    print(f"\n{len(R)} variants. SPY Sharpe dev {sp_d:.2f} / holdout {sp_h:.2f}.")
    print(f"beat SPY's CAGR: dev {(R.dev_vsSPY > 0).mean():.0%}, holdout {(R.hold_vsSPY > 0).mean():.0%}, both {((R.dev_vsSPY > 0) & (R.hold_vsSPY > 0)).mean():.0%}; "
          f"beat SPY's Sharpe: dev {(R.dev_Sharpe > sp_d).mean():.0%}, holdout {(R.hold_Sharpe > sp_h).mean():.0%}, "
          f"both {((R.dev_Sharpe > sp_d) & (R.hold_Sharpe > sp_h)).mean():.0%}")
    cols = ["W", "SPAN", "REBAL", "X", "dev_CAGR", "dev_Sharpe", "dev_vsSPY", "hold_CAGR", "hold_Sharpe", "hold_vsSPY", "all_CAGR", "all_DD"]
    print("\nTOP 15 PICKED ON 2019-22 SHARPE, then read on 2023-26:\n" + R.sort_values("dev_Sharpe", ascending=False)[cols].head(15).round(2).to_string(index=False))
    print("\nTOP 10 ON 2023-26 (hindsight -- for reference only):\n" + R.sort_values("hold_Sharpe", ascending=False)[cols].head(10).round(2).to_string(index=False))
    for k in ("X", "W", "SPAN", "REBAL"):
        print(f"\nAVERAGE BY {k}:\n" + R.groupby(k)[["dev_CAGR", "dev_Sharpe", "hold_CAGR", "hold_Sharpe", "all_DD"]].mean().round(2).to_string())
    b = R.sort_values("dev_Sharpe", ascending=False).iloc[0]
    print(f"::notice::growth+yield grid: {len(R)} variants; best on 2019-22 (W{b.W:.0f}/SPAN{b.SPAN:.0f}/REB{b.REBAL:.0f}/X{b.X:.0f}) "
          f"dev Sharpe {b.dev_Sharpe:.2f} -> 2023-26 {b.hold_Sharpe:.2f} (SPY {sp_d:.2f} / {sp_h:.2f})")


if __name__ == "__main__":
    main()
