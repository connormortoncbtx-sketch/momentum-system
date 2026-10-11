"""
research/earnings_week.py -- earnings-announcement-week premium (Oct 2026)

Do stocks earn more in weeks they report earnings, and does the live rule that
excludes earnings in the holding week cost the momentum basket?
Holding window = Monday close -> Friday close of week t+1 (designed timing).
A stock "reports in the window" if its reaction day d0 is Tue-Fri of week t+1
(a Monday d0 is already in the Monday close). Uses actual dates; in practice
dates are announced weeks ahead, so this is a mild look-ahead on timing only.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

P = os.environ.get("PANEL", "research_data")
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
U = B.universe(p, f, 5.0, 2e6)
fwd = B.next_week_returns(p, "designed")
start, split = pd.Timestamp("2017-01-06"), pd.Timestamp("2024-01-01")

e = pd.read_parquet(f"{P}/earnings_events.parquet")
e["d0"] = pd.to_datetime(e.d0)
# week label = that week's Friday; formation week = previous Friday
e["wk"] = e.d0 + pd.to_timedelta((4 - e.d0.dt.weekday) % 7, unit="D")
e = e[e.d0.dt.weekday >= 1]                         # Tue-Fri reaction days only
e["form"] = e.wk - pd.Timedelta(days=7)
ev = pd.DataFrame(False, index=p.weeks, columns=p.symbols)
ii = p.weeks.get_indexer(e.form); jj = p.symbols.get_indexer(e.symbol)
ok = (ii >= 0) & (jj >= 0)
arr = np.zeros(ev.shape, dtype=bool); arr[ii[ok], jj[ok]] = True
ev = pd.DataFrame(arr, index=p.weeks, columns=p.symbols)

rk = lambda x: B.xrank(x, U)
proxy = (0.35 * rk(f["rs_live"]) + 0.25 * rk(f["trend"]) + 0.20 * rk(f["hi52"])) / 0.80
m = U & fwd.notna()
uni = fwd.where(m).mean(axis=1)


def nw(x, lag=0):
    x = x.dropna().to_numpy(); n = len(x); e_ = x - x.mean(); v = e_ @ e_ / n
    for l in range(1, lag + 1):
        v += 2 * (1 - l / (lag + 1)) * (e_[l:] @ e_[:-l]) / n
    return x.mean() / np.sqrt(v / n)


def report(label, series):
    out = []
    for nm, sl in (("dev 2017-23", (series.index >= start) & (series.index < split)),
                   ("holdout 2024-26", series.index >= split)):
        s = series[sl].dropna()
        out.append(f"{nm}: {s.mean()*100:+.3f}%/wk (t {nw(s):+.2f}, {len(s)} wks)")
    print(f"{label:58s} " + " | ".join(out))


print("Excess return vs equal-weight universe, Mon close -> Fri close\n")
report("All stocks reporting this week", fwd.where(m & ev).mean(axis=1) - uni)
report("All stocks NOT reporting", fwd.where(m & ~ev).mean(axis=1) - uni)
for name, lo in (("top 10% momentum proxy", 0.9), ("top 2% momentum proxy (~top 50)", 0.98)):
    top = m & (proxy.where(m).rank(axis=1, pct=True) > lo)
    report(f"{name}, reporting", fwd.where(top & ev).mean(axis=1) - uni)
    report(f"{name}, not reporting", fwd.where(top & ~ev).mean(axis=1) - uni)
big = m & (f["adv"].where(U).rank(axis=1, ascending=False) <= 500)
report("Top-500 by $ volume, reporting", fwd.where(big & ev).mean(axis=1) - fwd.where(big).mean(axis=1))
small = m & ~big
report("Rest (smaller names), reporting", fwd.where(small & ev).mean(axis=1) - fwd.where(small).mean(axis=1))
print("\nShare of universe reporting in a given week (median): "
      f"{(ev & m).sum(axis=1).div(m.sum(axis=1)).median()*100:.1f}%")
# dispersion: how much riskier is an earnings week for one stock?
print("Cross-sectional sd of weekly return, reporting vs not: "
      f"{fwd.where(m & ev).std(axis=1).median()*100:.1f}% vs {fwd.where(m & ~ev).std(axis=1).median()*100:.1f}%")

# Portfolio: top-20 by proxy each week, with vs without the earnings exclusion
def basket(excl, n=20, cost=15.0):
    s = proxy.where(m & (~ev if excl else True))
    out = {}
    for wk, row in s.iterrows():
        row = row.dropna()
        if len(row) < n:
            continue
        out[wk] = fwd.loc[wk, row.nlargest(n).index].mean() - 2 * cost / 1e4
    return pd.Series(out)
for excl in (True, False):
    r = basket(excl); r = r[r.index >= start]
    for nm, sl in (("dev", r.index < split), ("holdout", r.index >= split)):
        st = B.stats(r[sl])
        print(f"top-20 weekly, {'EXCLUDE earnings' if excl else 'INCLUDE earnings'} {nm}: "
              f"CAGR {st['CAGR%']:.1f}% Sharpe {st['Sharpe']:.2f} maxDD {st['maxDD%']:.0f}%")
