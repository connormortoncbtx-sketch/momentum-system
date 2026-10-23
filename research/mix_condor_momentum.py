"""
research/mix_condor_momentum.py -- how much condor overlay on top of the momentum book? (Oct 23, 2026)

Momentum book: the live-system proxy (LIVE composite: 0.35 RS + 0.25 trend + 0.20 52-week-high, research/alpha_v2.py),
top 10 each Friday close, equal weight, held one week, 15 bp per side, ABSOLUTE weekly return (fully invested).
Condor book: the backtest's chosen same-day iron condor (research/options_premium*.py): k = 1 implied move, SPY $5 /
QQQ 0.8% wings, entered 10:00 ET, held to the close; return on risk per day at $0.01 and $0.03 per leg.
Overlay (a margin account): momentum stays 100% invested; the condors risk f of equity per day, split SPY/QQQ (the
COMBO book) -- their collateral is the max loss, which the stock positions' spare buying power covers.
Overlap Feb 2024 - Oct 2026 (~2.7 years; condor data starts Feb 2024). Reports, for f = 0 .. 10%: CAGR, volatility,
Sharpe, max drawdown, worst week; correlation of the two books (all weeks, and in momentum's worst weeks).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import alpha_v2 as A

RD = A.RD
pd.set_option("display.width", 220)


def momentum_weekly():
    p, U, R, live, ex1, ex4 = A.build()
    C = p.df("close")
    nxt = (C.shift(-1) / C - 1).clip(-0.9, 3)
    s = live.where(U)
    out, spy = {}, {}
    weeks = list(s.index)
    for i, w in enumerate(weeks[:-1]):
        if s.loc[w].notna().sum() < 50:
            continue
        picks = list(s.loc[w].dropna().nlargest(10).index)
        r = nxt.loc[w, picks].mean()
        if np.isfinite(r):
            out[weeks[i + 1]] = r - 2 * 0.0015                     # labelled by the Friday the week ENDS
        if "SPY" in C.columns and np.isfinite(nxt.loc[w, "SPY"]):
            spy[weeks[i + 1]] = nxt.loc[w, "SPY"]
    return pd.Series(out, name="mom"), pd.Series(spy, name="spy")


def condor_daily(cost):
    rows = []
    for und, f, W in (("SPY", "options_premium_trades.parquet", 5), ("QQQ", "options_premium_trades_QQQ.parquet", "0.8%")):
        t = pd.read_parquet(f"{RD}/results/{f}")
        t = t[(t.tenor == "0DTE") & (t.struct == "iron condor") & (t.k == 1.0) & (t.W.astype(str) == str(W)) & (t.cost == cost)]
        rows.append(t.set_index(pd.to_datetime(t.d)).ror.rename(und))
    d = pd.concat(rows, axis=1).sort_index()
    return d.mean(axis=1, skipna=True)                          # COMBO: half the risk on each, or all on the one that traded


def stats(r, label):
    nav = (1 + r).cumprod()
    dd = (nav / nav.cummax() - 1).min()
    yrs = len(r) / 52
    return dict(book=label, cagr_pct=(nav.iloc[-1] ** (1 / yrs) - 1) * 100, vol_pct=r.std() * np.sqrt(52) * 100,
                sharpe=r.mean() / r.std() * np.sqrt(52), max_dd_pct=dd * 100, worst_week_pct=r.min() * 100,
                up_weeks_pct=(r > 0).mean() * 100)


def main():
    mom, spy = momentum_weekly()
    res = []
    for cost in (0.01, 0.03):
        cd = condor_daily(cost)
        wk_end = pd.Series(mom.index, index=mom.index)
        # map each condor day to the Friday ending its week
        idx = np.searchsorted(mom.index.values, cd.index.values.astype("datetime64[ns]"), side="left")
        ok = idx < len(mom.index)
        cdw = pd.Series(cd.values[ok], index=mom.index[idx[ok]])
        both = mom.index[(mom.index >= cd.index.min()) & (mom.index <= cd.index.max() + pd.Timedelta("6D"))]
        m = mom.reindex(both)
        for f in (0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10):
            c = cdw.groupby(level=0).apply(lambda x: np.prod(1 + f * x) - 1).reindex(both).fillna(0)
            res.append({**stats(m + c, f"momentum + condors {f:.1%}/day"), "cost": cost, "f": f})
        c1 = cdw.groupby(level=0).apply(lambda x: np.prod(1 + 0.02 * x) - 1).reindex(both).fillna(0)
        res.append({**stats(c1, "condors alone 2%/day"), "cost": cost, "f": None})
        corr = np.corrcoef(m, c1)[0, 1]
        bad = m <= m.quantile(0.1)
        print(f"\ncost ${cost:.2f}/leg, {len(both)} weeks {both.min():%Y-%m-%d} - {both.max():%Y-%m-%d}: "
              f"weekly correlation momentum vs condors {corr:+.2f}; in momentum's worst 10% of weeks "
              f"(avg {m[bad].mean() * 100:+.1f}%) the condors made {c1[bad].mean() * 100:+.2f}% (2%/day) vs "
              f"{c1[~bad].mean() * 100:+.2f}% otherwise")
        if len(spy):
            s = spy.reindex(both).dropna()
            print(f"  momentum beta to SPY {np.cov(m.reindex(s.index), s)[0, 1] / s.var():.2f}; condor corr with SPY "
                  f"{np.corrcoef(c1.reindex(s.index), s)[0, 1]:+.2f}, with |SPY| {np.corrcoef(c1.reindex(s.index), s.abs())[0, 1]:+.2f}")
    R = pd.DataFrame(res)
    print("\n" + R.round(2).to_string(index=False))
    out = f"{RD}/results"
    os.makedirs(out, exist_ok=True)
    R.to_csv(f"{out}/mix_condor_momentum.csv", index=False)
    for cost in (0.01, 0.03):
        x = R[(R.cost == cost) & R.f.notna()]
        b = x.loc[x.sharpe.idxmax()]
        z = x[x.f == 0].iloc[0]
        print(f"::notice::mix (cost {cost}): momentum alone Sharpe {z.sharpe:.2f} CAGR {z.cagr_pct:.0f}% DD {z.max_dd_pct:.0f}%; "
              f"best overlay {b.f:.1%}/day -> Sharpe {b.sharpe:.2f} CAGR {b.cagr_pct:.0f}% DD {b.max_dd_pct:.0f}%; "
              f"2%/day -> Sharpe {x[x.f == 0.02].sharpe.iloc[0]:.2f} CAGR {x[x.f == 0.02].cagr_pct.iloc[0]:.0f}% DD {x[x.f == 0.02].max_dd_pct.iloc[0]:.0f}%")


if __name__ == "__main__":
    main()
