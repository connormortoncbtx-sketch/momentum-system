"""
research/loser_rebound.py -- "so bad they're good" (Nov 15, 2026; Connor: low-scored picks often did great, top-scored
ones looked priced out). Monthly panel and universe of novel_eval.py ($5, $2M/day, no funds, delisted included,
15 bps/side). Dev 2018-23, holdout 2024-26.

PRE-REGISTERED (before looking):
 A  CONCENTRATION (descriptive): by 12-1 momentum decile -- next-month mean / median return, P(>= +20%), P(<= -20%),
    vol, and each decile's equal-weight Sharpe. Answers "how often do the worst-scored names do great".
 B  LONG-TERM REVERSAL (De Bondt & Thaler): score = return over months t-36..t-13. PASS if the bottom decile beats the
    equal-weight universe by t > 2 in BOTH periods AND its long-only book (net) has a higher Sharpe than the universe in
    both periods.
 C  LOSER REBOUND AFTER STRESS (Daniel & Moskowitz momentum crashes): bottom 12-1 decile minus universe, next month,
    split by market state at formation: universe EW index >= 15% below its 12-month high vs not. Reported only (few
    stress months: 2018Q4, 2020, 2022).
 D  CONTROL: 1-month losers (bottom decile of last month's return) vs universe -- the short-term reversal already found
    to have decayed in large caps (reversal_lc.py).
 E  OVERPRICED TOP: top 12-1 decile split by last month's return (top third = "spiked") -- does the spiked part lag?
    Reported only (the earlier rank-band study found reversal in the top ranks' last-week spike).
Delisting: a name in the universe whose next-month price is missing gets -30% (sensitivity: 0%) in the loser books.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import novel_eval as E  # noqa: E402
import novel_stock_signals as N  # noqa: E402

RD = E.RD
PER = E.PER


def tstat(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 5 else np.nan


def book_sel(sel, fwd, univ, dl):
    """Equal-weight monthly book of boolean selection `sel`, net of turnover cost; missing next price -> dl."""
    out, prev = {}, set()
    for m in sel.index:
        pick = set(sel.columns[sel.loc[m].fillna(False).to_numpy()])
        if len(pick) < 10:
            continue
        r = fwd.loc[m].reindex(list(pick)).fillna(dl).mean()
        out[m] = r - len(pick ^ prev) / len(pick) * N.COST
        prev = pick
    return pd.Series(out)


def main():
    D, C, ret, fwd, univ, m12 = E.monthly_panel()
    U = univ.astype(bool)
    rows = []
    ew = fwd.where(U).mean(axis=1)
    rk12 = m12.where(U).rank(axis=1, pct=True)
    dec = np.ceil(rk12 * 10).clip(1, 10)

    # A: concentration by 12-1 decile
    for lab, a, z in PER:
        for d in range(1, 11):
            f = fwd.loc[a:z].where(dec.loc[a:z] == d).stack().dropna()
            sel = dec.loc[a:z] == d
            b = book_sel(sel, fwd, univ, -0.30)
            rows.append(dict(test="A 12-1 decile", period=lab, bucket=d, mean_pct=f.mean() * 100, median_pct=f.median() * 100,
                             p_up20=(f >= 0.2).mean() * 100, p_dn20=(f <= -0.2).mean() * 100, vol_pct=f.std() * 100,
                             book_sharpe=N.stats(b)["sharpe"], book_cagr=N.stats(b)["cagr"]))
        u = N.stats(ew.loc[a:z])
        rows.append(dict(test="A universe EW", period=lab, book_sharpe=u["sharpe"], book_cagr=u["cagr"]))

    # B: long-term reversal; D: 1-month reversal
    lt = C.shift(13) / C.shift(36) - 1
    for name, sc in (("B long-term losers (t-36..t-13)", lt), ("D 1-month losers", ret)):
        r = sc.where(U).rank(axis=1, pct=True)
        sel = r <= 0.1
        ex = pd.Series({m: fwd.loc[m][sel.loc[m]].fillna(-0.30).mean() - ew.loc[m] for m in sel.index if sel.loc[m].sum() >= 10})
        for dl in (-0.30, 0.0):
            b = book_sel(sel, fwd, univ, dl)
            for lab, a, z in PER:
                x = ex.loc[a:z]
                rows.append(dict(test=name, period=lab, delist=dl, excess_bp=x.mean() * 1e4, t=tstat(x), n=len(x),
                                 book_sharpe=N.stats(b.loc[a:z])["sharpe"], book_cagr=N.stats(b.loc[a:z])["cagr"],
                                 univ_sharpe=N.stats(ew.loc[a:z])["sharpe"]))

    # C: loser rebound after stress
    idx = (1 + ret.where(U).mean(axis=1).fillna(0)).cumprod()
    dd = idx / idx.rolling(12, min_periods=6).max() - 1
    lose = (rk12 <= 0.1)
    lx = pd.Series({m: fwd.loc[m][lose.loc[m]].fillna(-0.30).mean() - ew.loc[m] for m in lose.index if lose.loc[m].sum() >= 10})
    for lab, a, z in PER + (("2018-26", "2018-01", "2026-12"),):
        x, s = lx.loc[a:z], dd.reindex(lx.loc[a:z].index) <= -0.15
        rows.append(dict(test="C 12-1 losers - universe | stress", period=lab, excess_bp=x[s].mean() * 1e4, t=tstat(x[s]), n=int(s.sum())))
        rows.append(dict(test="C 12-1 losers - universe | normal", period=lab, excess_bp=x[~s].mean() * 1e4, t=tstat(x[~s]), n=int((~s).sum())))

    # E: spiked vs not within the top 12-1 decile
    top = rk12 >= 0.9
    r1 = ret.where(top).rank(axis=1, pct=True)
    sp = pd.Series({m: fwd.loc[m][r1.loc[m] >= 2 / 3].mean() - fwd.loc[m][r1.loc[m] < 2 / 3].mean() for m in r1.index
                    if top.loc[m].sum() >= 30})
    for lab, a, z in PER:
        x = sp.loc[a:z]
        rows.append(dict(test="E top decile: spiked third - rest", period=lab, excess_bp=x.mean() * 1e4, t=tstat(x), n=len(x)))

    R = pd.DataFrame(rows)
    R.to_csv(f"{RD}/results/loser_rebound.csv", index=False)
    print(R.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
