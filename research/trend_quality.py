"""
research/trend_quality.py -- does the QUALITY of a trend tell good momentum stocks from bad ones? (Nov 13, 2026;
"more sophisticated methods", step 3.)

Among momentum stocks (top 20% by 12-1 month return in the universe each Friday), sort by trend-quality measures from
daily prices, all computed from data up to that Friday:
  FIP     "frog in the pan" information discreteness (Da, Gurun & Warachka 2014): sign(12-1 return) x (% down days -
          % up days) over the same 12-1 window. LOW = many small steady gains (continuous) -- published to beat
          momentum built from a few jumps.
  R2      R-squared of log price on time over the last 126 trading days (smoothness of the climb)
  VR5     variance ratio of 5-day vs 1-day returns over 252 days (> 1 = persistent / trending, a Hurst-type measure)
  VOLCLU  20-day volatility / 126-day volatility (calm now vs. its own history; < 1 = calming)
  DD126   worst drawdown inside the last 126 days (shallow = orderly)
For each: the best third vs the worst third of momentum stocks by next-4-week return (entry Monday close), and books:
top 10 / 30 by momentum within the best-quality half vs plain momentum top 10 / 30, rebalanced every 4 weeks with a
top-2x buffer, spread costs. Dev 2017-2023 / holdout 2024-Oct 2026 (holdout lacks most delistings). Universe price
>= $5, 20-day $ volume >= $2M, funds excluded. Public data only.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import universe_floor as U  # noqa: E402

RD = os.environ.get("RESEARCH_DATA", "research_data")
SPLIT = pd.Timestamp("2024-01-01")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)


def main():
    D = pd.concat([pd.read_parquet(f, columns=["symbol", "date", "close", "volume"]) for f in sorted(glob.glob(f"{RD}/daily/ohlcv_*.parquet"))])
    try:
        import backtest as B
        p = B.load_panel(RD)
        funds = {s for s, isf in zip(p.symbols, p.is_fund) if isf}
        D = D[~D.symbol.isin(funds)]
    except Exception as e:
        print("fund filter skipped:", e)
    C = D.pivot_table(index="date", columns="symbol", values="close").astype("float32")
    V = D.pivot_table(index="date", columns="symbol", values="volume").astype("float32")
    del D
    print(f"daily matrix {C.shape}", flush=True)
    r = C.pct_change(fill_method=None).where(lambda x: x.abs() < 0.7)
    lp = np.log(C)
    dv20 = (C * V).rolling(20, min_periods=15).mean()
    mom = C.shift(21) / C.shift(252) - 1
    sgn = np.sign(r)
    up = (sgn > 0).astype("float32").where(r.notna()).shift(21).rolling(231, min_periods=180).mean()
    dn = (sgn < 0).astype("float32").where(r.notna()).shift(21).rolling(231, min_periods=180).mean()
    fip = np.sign(mom) * (dn - up)
    t = pd.Series(np.arange(len(C), dtype="float64"), index=C.index)
    r2 = lp.rolling(126, min_periods=100).corr(t) ** 2
    r5 = lp - lp.shift(5)
    vr5 = r5.rolling(252, min_periods=200).var() / (5 * r.rolling(252, min_periods=200).var())
    volclu = r.rolling(20, min_periods=15).std() / r.rolling(126, min_periods=100).std()
    dd126 = (C / C.rolling(126, min_periods=100).max() - 1).rolling(126, min_periods=100).min()
    # Fridays (last trading day of each week)
    idx = C.index.to_series()
    fri = idx.groupby(idx.dt.to_period("W-FRI")).max().values
    fri = pd.DatetimeIndex(fri)
    fri = fri[fri >= "2017-01-01"]
    pos = C.index.get_indexer(fri)
    nxt = lambda k: C.shift(-k)
    f1 = (nxt(6) / nxt(1) - 1).iloc[pos]            # Monday close -> following Monday close
    f4 = (nxt(21) / nxt(1) - 1).iloc[pos]
    f1, f4 = f1.where(f1.abs() < 1.5), f4.where(f4.abs() < 3)
    take = lambda x: x.iloc[pos]
    px, adv = take(C), take(dv20)
    uni = (px >= 5) & (adv >= 2e6) & take(mom).notna()
    M = take(mom).where(uni)
    momtop = M.rank(axis=1, pct=True) >= 0.8
    Q = {"FIP (low=steady)": -take(fip), "R2 (smooth)": take(r2), "VR5 (persistent)": take(vr5),
         "VOLCLU (calming)": -take(volclu), "DD126 (shallow)": take(dd126)}
    ex4 = f4.sub(f4.where(uni).mean(axis=1), axis=0)
    rows = []
    for name, q in Q.items():
        qq = q.where(momtop)
        rk = qq.rank(axis=1, pct=True)
        best, worst = rk >= 2 / 3, rk <= 1 / 3
        spread = ex4.where(best).mean(axis=1) - ex4.where(worst).mean(axis=1)
        b_ex = ex4.where(best).mean(axis=1); w_ex = ex4.where(worst).mean(axis=1)
        row = dict(measure=name)
        for lab, sel in (("dev", spread.index < SPLIT), ("hold", spread.index >= SPLIT)):
            s4 = spread[sel].dropna()
            # 4-week returns overlap week to week: t-stat from non-overlapping (every 4th) observations
            s_no = s4.iloc[::4]
            row[f"{lab}_best_minus_worst_4w%"] = s4.mean() * 100
            row[f"{lab}_t"] = s_no.mean() / s_no.std() * np.sqrt(len(s_no))
            row[f"{lab}_best_vs_univ_4w%"] = b_ex[sel].mean() * 100
            row[f"{lab}_worst_vs_univ_4w%"] = w_ex[sel].mean() * 100
        rows.append(row)
    R = pd.DataFrame(rows)
    print("\nAMONG MOMENTUM STOCKS (top 20% by 12-1): best third vs worst third on each trend-quality measure,\n"
          "next-4-week return in excess of the universe average, % (t from non-overlapping 4-week periods):")
    print(R.round(2).to_string(index=False))
    # books
    small = (px < 5) | (adv < 2e6)
    cost = pd.DataFrame(U.half_spread_bps(px.fillna(0).to_numpy(), adv.fillna(0).to_numpy()), index=px.index, columns=px.columns)
    wr = f1
    books = []
    variants = {"plain 12-1 momentum": M}
    for name, q in Q.items():
        good = q.where(momtop).rank(axis=1, pct=True) >= 0.5
        variants[f"12-1 within best-half {name.split()[0]}"] = M.where(good | ~momtop.fillna(False)).where(momtop)
    for vn, s in variants.items():
        for n in (10, 30):
            r_ = U.sim(s, uni, wr, cost, small, n=n, k=4, buffer=2 * n)
            d, h = U.st(r_[r_.index < SPLIT]), U.st(r_[r_.index >= SPLIT])
            books.append(dict(book=vn, n=n, dev_CAGR=d["CAGR"], dev_Sharpe=d["Sharpe"], dev_DD=d["maxDD"],
                              hold_CAGR=h["CAGR"], hold_Sharpe=h["Sharpe"], hold_DD=h["maxDD"]))
    Bk = pd.DataFrame(books)
    print("\nBOOKS (rebalanced every 4 weeks, top-2x buffer, after spread costs):")
    print(Bk.round(2).to_string(index=False))
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/trend_quality_sorts.csv", index=False)
    Bk.to_csv(f"{RD}/results/trend_quality_books.csv", index=False)
    print("::notice::trend quality (best-worst third, 4w %, dev/hold t): " + "; ".join(
        f"{r.measure.split()[0]} {r['dev_best_minus_worst_4w%']:+.2f}/{r['hold_best_minus_worst_4w%']:+.2f} "
        f"(t {r.dev_t:.1f}/{r.hold_t:.1f})" for _, r in R.iterrows()))


if __name__ == "__main__":
    main()
