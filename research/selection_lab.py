"""
research/selection_lab.py -- picking better stocks (Nov 13, 2026; Connor: "we need to make it so we are picking the
right stocks").

What the data says so far: WHICH momentum works changes with the market. Last 52 weeks: old weekly pick-10 (short-term
blend) +66%, 12-1 momentum -25..-33%; 2017-23 the reverse. So test documented fixes for exactly that:
  A. RESIDUAL momentum (Blitz, Huij & Martens 2011): rank on the stock's own trend after removing its market beta,
     scaled by its residual volatility -- less exposed to market turns, historically about half the crash risk.
  B. HORIZON SWITCHING (factor momentum, Ehsani & Linnainmaa 2022; Gupta & Kelly 2019): each week use the signal whose
     own top-10 book did best over the trailing L weeks (L = 13, 26, 52), using only returns already known.
  C. ENSEMBLE: equal rank-average of all candidate signals.
  D. CRASH GUARD (Daniel & Moskowitz 2016): momentum crashes come after bear markets when the market rebounds; hold
     cash when the equal-weight universe's trailing 2-year return is negative AND its 13-week vol is above its median.
Universe: today's U5_2M (price >= $5, ADV >= $2M; see universe_floor.py). Costs: per-name half-spread table, 1x.
Books: weekly pick-10 (full turnover) and monthly top-10 with a top-30 buffer.
Bar for any change: beat BOTH incumbents (live proxy = old pick-10, m12_1 = SLOW sleeve) on Sharpe in dev
(2017-23) AND holdout (2024-Sep 2026), not just one. Public data only -> results/selection_lab.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as B  # noqa: E402
import universe_floor as U  # noqa: E402

RD = os.environ.get("RESEARCH_DATA", "research_data")
SPLIT = pd.Timestamp("2024-01-01")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)


def main():
    p = B.load_panel(RD); f = B.features(p)
    base = (f["hist"] >= 53) & ~pd.DataFrame(np.tile(p.is_fund, (len(p.weeks), 1)), index=p.weeks, columns=p.symbols) \
        & ~f["rsplit_52w"].fillna(False).astype(bool)
    px, adv = f["price"], f["adv"]
    m = base & (px >= 5) & (adv >= 2e6)
    small = (px < 5) | (adv < 2e6)
    cost = pd.DataFrame(U.half_spread_bps(px.fillna(0).to_numpy(), adv.fillna(0).to_numpy()), index=p.weeks, columns=p.symbols)
    d1c = p.df("d1_close")
    wr = (d1c.shift(-2) / d1c.shift(-1) - 1); wr = wr.where(wr.abs() < 3)
    rk = lambda x: x.where(m).rank(axis=1, pct=True)

    # residual momentum: weekly returns vs the equal-weight universe, rolling 52-week beta
    C = p.df("close"); r1 = (C / C.shift(1) - 1).where(lambda x: x.abs() < 1.5)
    mkt = r1.where(m).mean(axis=1)
    W = 52
    mx = mkt.rolling(W, min_periods=40)
    cov = (r1.mul(mkt, axis=0)).rolling(W, min_periods=40).mean() - r1.rolling(W, min_periods=40).mean().mul(mx.mean(), axis=0)
    beta = cov.div(mx.var(ddof=0), axis=0)
    eps = r1 - beta.shift(1).mul(mkt, axis=0)                    # beta from data up to t-1
    res12 = eps.shift(4).rolling(48, min_periods=36).sum() / eps.shift(4).rolling(48, min_periods=36).std()
    res6 = eps.shift(4).rolling(22, min_periods=18).sum() / eps.shift(4).rolling(22, min_periods=18).std()

    sig = {
        "live_proxy": (0.35 * rk(f["rs_live"]) + 0.25 * rk(f["trend"]) + 0.20 * rk(f["hi52"])) / 0.80,
        "m12_1": f["m12_1"].where(m),
        "m6_1": f["m6_1"].where(m),
        "m3_1": f["m3_1"].where(m),
        "r4w": f["r4w"].where(m),
        "hi52": f["hi52"].where(m),
        "resid12": res12.where(m),
        "resid6": res6.where(m),
    }
    sig["ensemble"] = sum(rk(s) for s in sig.values()) / len(sig)

    def run(score, k, buf):
        return U.sim(score, m, wr, cost, small, k=k, buffer=buf)

    books = {"weekly_pick10": (1, None), "monthly_buf30": (4, 30)}
    series = {}
    for sn, s in sig.items():
        for bn, (k, b) in books.items():
            series[(sn, bn)] = run(s, k, b)
        print(f"  {sn}", flush=True)

    # B. horizon switching on the weekly pick-10 track records (known at decision time: shift 1)
    cands = ["live_proxy", "m12_1", "m6_1", "m3_1", "r4w", "hi52", "resid12", "resid6"]
    track = pd.DataFrame({c: series[(c, "weekly_pick10")] for c in cands}).reindex(p.weeks)
    for L in (13, 26, 52):
        perf = np.log1p(track.fillna(0)).rolling(L, min_periods=L).sum().shift(1)
        best = perf.dropna(how="all").idxmax(axis=1)
        sc = pd.DataFrame(np.nan, index=p.weeks, columns=p.symbols)
        for c in cands:
            wk = best.index[best == c]
            sc.loc[wk] = rk(sig[c]).loc[wk]
        sig[f"switch{L}"] = sc
        for bn, (k, b) in books.items():
            series[(f"switch{L}", bn)] = run(sc, k, b)
        print(f"  switch{L}: picks " + best.value_counts().head(4).to_dict().__str__(), flush=True)

    # D. crash guard overlay on every series
    ew = mkt.fillna(0)
    two_yr = np.log1p(ew).rolling(104, min_periods=80).sum()
    vol13 = ew.rolling(13).std()
    danger = ((two_yr < 0) & (vol13 > vol13.expanding(52).median())).shift(0)   # known at formation week t
    print(f"crash-guard weeks in cash: {int(danger.sum())} of {len(danger)} "
          f"({', '.join(sorted({str(d.year) for d in danger[danger].index}))})", flush=True)
    for key in list(series):
        r = series[key]
        series[(key[0] + "+guard", key[1])] = r.where(~danger.reindex(r.index).fillna(False), 0.0)

    rows = []
    for (sn, bn), r in series.items():
        d, h = U.st(r[r.index < SPLIT]), U.st(r[r.index >= SPLIT])
        l52 = U.st(r[r.index >= r.index[-1] - pd.Timedelta(weeks=52)])
        rows.append(dict(signal=sn, book=bn, dev_CAGR=d["CAGR"], dev_Sharpe=d["Sharpe"], dev_DD=d["maxDD"],
                         hold_CAGR=h["CAGR"], hold_Sharpe=h["Sharpe"], hold_DD=h["maxDD"], last52_CAGR=l52["CAGR"]))
    R = pd.DataFrame(rows)
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/selection_lab.csv", index=False)
    for bn in books:
        x = R[R.book == bn].sort_values("hold_Sharpe", ascending=False)
        inc = x[x.signal.isin(["live_proxy", "m12_1"])]
        bar_d, bar_h = inc.dev_Sharpe.max(), inc.hold_Sharpe.max()
        x = x.assign(beats_both=(x.dev_Sharpe > bar_d) & (x.hold_Sharpe > bar_h))
        print(f"\n{bn.upper()} (bar: dev Sharpe > {bar_d:.2f} AND holdout Sharpe > {bar_h:.2f})\n"
              + x.drop(columns="book").round(2).to_string(index=False))
    yr = pd.DataFrame({k[0]: v.groupby(v.index.year).apply(lambda z: ((1 + z).prod() - 1) * 100)
                       for k, v in series.items() if k[1] == "monthly_buf30" and "+guard" not in k[0]})
    print("\nBY YEAR, monthly top-10, % return:\n" + yr.round(1).to_string())
    win = R[(R.dev_Sharpe > 0) & (R.hold_Sharpe > 0)].sort_values("dev_Sharpe").tail(3)
    print("::notice::selection lab: best by dev Sharpe -> " + "; ".join(
        f"{r.signal}/{r.book} dev {r.dev_Sharpe:.2f} hold {r.hold_Sharpe:.2f}" for r in win.itertuples()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
