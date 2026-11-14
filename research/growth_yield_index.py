"""
research/growth_yield_index.py -- "growth AND yield" index (Nov 13, 2026; Connor): hold the X stocks that most
consistently beat the S&P 500 on BOTH price growth and dividend yield over rolling windows; prune the ones that stop.

Measures (weekly panel; `close` is split+dividend adjusted, `raw_close` is not, so their ratio drifts by each dividend):
  DY(t, W)   dividend yield over the W weeks to t  = sum of weekly log changes in close/raw_close (|change| > 15% treated
             as a split and ignored)
  PG(t, W)   price growth = total log return - DY
  beat(t)    PG > SPY's PG AND DY > SPY's DY over the 52 weeks to t
Score at t (variants):
  CONSIST3y  share of the 9 rolling 52-week windows ending every 13 weeks over the last 2 years that 'beat' (ties broken
             by average excess total return) -- "continuously accomplishes it"
  BOTH156    beat on both over a single 3-year window, ranked by the smaller of the two margins
Contrast books: YIELD (top X by 52w yield only), GROWTH (top X by 52w price growth only), and SPY.
Index: top X (20 / 30 / 50) equal weight, rebalanced every 13 weeks, a holding is kept while it ranks within 2X
(prune when it falls out). Universe: price >= $5, ADV >= $5M, non-fund, >= 3 years listed. Spread costs as in
universe_floor.py. Delistings: −30% if small, else 0%.
Periods: holdings start 2019 (3 years of history needed); 2019-2022 and 2023-Sep 2026. Survivorship caveat: the panel
lacks most 2023-26 delistings (being fixed); large dividend payers are the least affected. Public data only.
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
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)


def main():
    p = B.load_panel(RD); f = B.features(p)
    C, Rw = p.df("close"), p.df("raw_close")
    fac = np.log(C / Rw)
    dfac = fac.diff().where(lambda x: x.abs() <= 0.15, 0.0).clip(lower=0)      # dividends only
    tr = np.log(C / C.shift(1)).where(lambda x: x.abs() < 1.5)
    DY = lambda W: dfac.rolling(W, min_periods=int(W * 0.9)).sum()
    TR = lambda W: tr.rolling(W, min_periods=int(W * 0.9)).sum()
    dy52, tr52 = DY(52), TR(52); pg52 = tr52 - dy52
    dy156, tr156 = DY(156), TR(156); pg156 = tr156 - dy156
    spy = "SPY"
    beat = (pg52.sub(pg52[spy], axis=0) > 0) & (dy52.sub(dy52[spy], axis=0) > 0)
    exc = tr52.sub(tr52[spy], axis=0)
    lags = [0, 13, 26, 39, 52, 65, 78, 91, 104]
    consist = sum(beat.shift(l).astype(float) for l in lags) / len(lags)
    avgexc = sum(exc.shift(l) for l in lags) / len(lags)
    sc_cons = consist + 0.001 * avgexc.rank(axis=1, pct=True)
    m_pg, m_dy = pg156.sub(pg156[spy], axis=0), dy156.sub(dy156[spy], axis=0)
    sc_both = np.minimum(m_pg / m_pg.std(axis=1).values[:, None], m_dy / m_dy.std(axis=1).values[:, None])
    sc_both = sc_both.where((m_pg > 0) & (m_dy > 0))

    base = (C.notna().cumsum() >= 156) & ~pd.DataFrame(np.tile(p.is_fund, (len(p.weeks), 1)), index=p.weeks, columns=p.symbols)
    m = base & (f["price"] >= 5) & (f["adv"] >= 5e6)
    small = (f["price"] < 5) | (f["adv"] < 2e6)
    cost = pd.DataFrame(U.half_spread_bps(f["price"].fillna(0).to_numpy(), f["adv"].fillna(0).to_numpy()),
                        index=p.weeks, columns=p.symbols)
    d1c = p.df("d1_close"); wr = (d1c.shift(-2) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 1.5)
    start = pd.Timestamp("2019-01-04")
    scores = {"CONSIST3y": sc_cons, "BOTH156": sc_both, "YIELD": dy52, "GROWTH": pg52}
    spy_r = wr[spy]
    rows, series = [], {}
    for sn, s in scores.items():
        s = s.where(m)
        if sn == "CONSIST3y":
            s = s.where(consist > 0)          # must have beaten in at least one window
        s = s.loc[s.index >= start]
        for X in (20, 30, 50):
            r = U.sim(s, m.loc[s.index], wr.loc[s.index], cost.loc[s.index], small.loc[s.index], n=X, k=13, buffer=2 * X)
            series[(sn, X)] = r
            for lab, rr in (("2019-22", r[r.index < SPLIT]), ("2023-26", r[r.index >= SPLIT]), ("all", r)):
                st = U.st(rr); sp = U.st(spy_r.reindex(rr.index))
                j = pd.concat([rr, spy_r.reindex(rr.index)], axis=1).dropna()
                beta = np.polyfit(j.iloc[:, 1], j.iloc[:, 0], 1)[0] if len(j) > 20 else np.nan
                rows.append(dict(book=sn, X=X, period=lab, CAGR=st["CAGR"], Sharpe=st["Sharpe"], maxDD=st["maxDD"],
                                 SPY_CAGR=sp["CAGR"], SPY_Sharpe=sp["Sharpe"], beta=beta))
        print(f"  {sn}", flush=True)
    R = pd.DataFrame(rows)
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/growth_yield_index.csv", index=False)
    print(R.round(2).to_string(index=False))
    yr = pd.DataFrame({f"{k[0]}_{k[1]}": v.groupby(v.index.year).apply(lambda z: ((1 + z).prod() - 1) * 100)
                       for k, v in series.items() if k[1] == 30})
    yr["SPY"] = spy_r[spy_r.index >= start].groupby(spy_r[spy_r.index >= start].index.year).apply(lambda z: ((1 + z).prod() - 1) * 100)
    print("\nBY YEAR (X=30), %:\n" + yr.round(1).to_string())
    # what it holds now
    last = sc_cons.where(m).iloc[-1].dropna().sort_values(ascending=False)
    print("\nCONSIST3y top 30 today:", ", ".join(f"{s}({consist.iloc[-1][s]:.0%})" for s in last.index[:30]))
    print(f"stocks beating SPY on both in the latest 52 weeks: {int(beat.iloc[-1][m.iloc[-1]].sum())} of {int(m.iloc[-1].sum())}; "
          f"in all 9 windows: {int((consist.iloc[-1][m.iloc[-1]] == 1).sum())}")
    a = R[(R.period == "all") & (R.X == 30)]
    print("::notice::growth+yield index (X=30, 2019-26 CAGR/Sharpe vs SPY): " +
          "; ".join(f"{r.book} {r.CAGR:+.1f}%/{r.Sharpe:.2f}" for r in a.itertuples()) + f"; SPY {a.SPY_CAGR.iloc[0]:+.1f}%/{a.SPY_Sharpe.iloc[0]:.2f}")


if __name__ == "__main__":
    main()
