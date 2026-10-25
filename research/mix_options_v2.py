"""
research/mix_options_v2.py -- which options sleeve, if any, belongs next to the momentum v2 book? (Oct 25, 2026)

Revisits mix_condor_v2.py (Feb 2024 - Oct 2026 only, morning condor) with the ThetaData series: 2018 - Oct 2026 daily
return on risk for same-day SPY/QQQ condors, put credit spreads and call credit spreads (natural fills; real quotes,
calibrated trade-bar proxy on the gap days) -- results/options_series_daily.parquet from options_theta_full.py series.

Books: STACK_v2 (shadow #7 v2, the momentum v2 bucket), and for context SLOW_h and LIVE50 (see mix_condor_v2.py).
Sleeves: each day risk f of equity per structure (SPY and QQQ half each when both trade), compounded into weeks.
Protocol (no peeking): DEV = 2018-2023, HOLDOUT = 2024 - Oct 2026, scored once.
  1. per structure, pick ONE variant (entry, shorts k, wing, exit) by DEV Sharpe of the sleeve alone
  2. pick the risk per structure f in {0, 1, 2, 3, 5}% (all 125 mixes) by DEV Sharpe of book + sleeves
  3. report book alone / + each sleeve at 2% / the dev-chosen mix, in DEV and HOLDOUT; correlations; how each sleeve
     did in the book's worst 10% of weeks (does it hedge, or pile on?)
"""
import io, contextlib, itertools, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import alpha_v2 as A
from mix_condor_momentum import stats
from mix_condor_v2 import live_books

pd.set_option("display.width", 250)
SPLIT, START = pd.Timestamp("2024-01-01"), pd.Timestamp("2018-01-01")
FS = (0.0, 0.01, 0.02, 0.03, 0.05)


def books():
    with contextlib.redirect_stdout(io.StringIO()):
        import hedged_stack as Hs
    st = (0.104 * Hs.H["SLOW_h"] + 0.238 * Hs.H["CONTRACT_h"] + 0.504 * Hs.R["ETFTREND"] + 0.154 * Hs.R["CRYPTO"].fillna(0)).dropna()
    lev = (0.020698 / st.rolling(8).std().shift(1)).clip(0.5, 3).fillna(1.0)
    B = pd.DataFrame({"STACK_v2": lev * st - (lev - 1).clip(lower=0) * 0.06 / 52, "SLOW_h": Hs.H["SLOW_h"]})
    _, l50 = live_books()
    l50.index = pd.DatetimeIndex(l50.index)
    B = B.join(l50, how="outer")
    B.index = pd.DatetimeIndex(B.index)
    return B[B.index >= START], Hs.mk["SPY"]


def sleeves():
    D = pd.read_parquet(f"{A.RD}/results/options_series_daily.parquet")
    D["d"] = pd.to_datetime(D.d)
    D = D.astype({c: str for c in ("entry", "struct", "exit", "und", "src")})
    key = ["struct", "entry", "k", "wing", "exit"]
    daily = D.groupby(key + ["d"]).ror.mean()                       # SPY + QQQ half each (or the one that traded)
    return daily, key


def weekly(r, f, idx):
    fri = r.index + pd.to_timedelta((4 - r.index.dayofweek) % 7, unit="D")
    w = (1 + f * r).groupby(fri).prod() - 1
    return w.reindex(idx).fillna(0)


def main():
    B, spy = books()
    daily, key = sleeves()
    idx = B.index[(B.index >= START) & (B.index <= daily.index.get_level_values("d").max() + pd.Timedelta("6D"))]
    dev, hold = idx < SPLIT, idx >= SPLIT
    # 1. one variant per structure, by dev Sharpe of the sleeve alone (2% risk/day)
    pick, cand, ser = {}, [], {}
    for v, r in daily.groupby(level=key):
        r = r.droplevel(key)
        ser[v] = r
        w = weekly(r, 0.02, idx)
        s_dev = w[dev].mean() / w[dev].std() * np.sqrt(52)
        s_hold = w[hold].mean() / w[hold].std() * np.sqrt(52)
        mb = B["STACK_v2"].reindex(idx).fillna(0) + w                 # the objective: the book WITH the sleeve
        mix_dev = mb[dev].mean() / mb[dev].std() * np.sqrt(52)
        cand.append(dict(zip(key, v), trades=len(r), mix_dev_sharpe=mix_dev, dev_sharpe=s_dev, holdout_sharpe=s_hold,
                         dev_ror_pct=r[r.index < SPLIT].mean() * 100, holdout_ror_pct=r[r.index >= SPLIT].mean() * 100))
    C = pd.DataFrame(cand)
    for st_, g in C.groupby("struct"):
        # pick by DEV Sharpe of STACK_v2 + this sleeve at 2%/day: with a fixed max-loss budget, a sleeve's own Sharpe
        # favours far, wide, low-return spreads that add little to the book (first run picked k1.5 / widest wings)
        best = g.sort_values("mix_dev_sharpe", ascending=False).iloc[0]
        pick[st_] = next(v for v in ser if v[0] == st_ and all(
            (abs(float(v[i]) - float(best[k])) < 1e-6) if k in ("k", "wing") else str(v[i]) == str(best[k]) for i, k in enumerate(key)))
        print(f"\n{st_}: top 8 variants by DEV Sharpe of STACK_v2 + sleeve (2%/day)\n"
              + g.sort_values("mix_dev_sharpe", ascending=False).head(8).round(2).to_string(index=False))
    C.to_csv(f"{A.RD}/results/mix_options_v2_sleeves.csv", index=False)
    S = {st_: ser[v] for st_, v in pick.items()}
    print(f"weeks in window {len(idx)} (dev {dev.sum()}, holdout {hold.sum()}); weeks with a sleeve trade: "
          + ", ".join(f"{k} {(weekly(r, 0.02, idx) != 0).sum()}" for k, r in S.items()))
    W = {st_: {f: weekly(r, f, idx) for f in FS} for st_, r in S.items()}

    res, corr = [], []
    for name in ("STACK_v2", "SLOW_h", "LIVE50"):
        m = B[name].reindex(idx)
        ok = m.notna()
        if ok.sum() < 100:
            continue
        mo = m.fillna(0)
        for st_ in S:
            w2 = W[st_][0.02]
            worst = mo[ok & dev] <= mo[ok & dev].quantile(0.1)
            corr.append(dict(book=name, sleeve=st_, corr_dev=np.corrcoef(mo[ok & dev], w2[ok & dev])[0, 1],
                             corr_hold=np.corrcoef(mo[ok & hold], w2[ok & hold])[0, 1],
                             corr_spy=np.corrcoef(spy.reindex(idx).fillna(0)[ok], w2[ok])[0, 1],
                             sleeve_in_book_worst10pct_dev=w2[ok & dev][worst].mean() * 100,
                             sleeve_other_weeks_dev=w2[ok & dev][~worst].mean() * 100))
        # 2. dev-chosen split of a fixed total option risk budget (1% steps). Sharpe alone would always say "more",
        #    since the sleeves are positive and nearly uncorrelated -- the budget is a drawdown choice, the split isn't.
        mixes = {"alone": dict.fromkeys(S, 0.0)}
        for st_ in S:
            mixes[f"+{st_} 2%"] = {**dict.fromkeys(S, 0.0), st_: 0.02}
        Wf = {st_: {f: weekly(S[st_], f, idx) for f in np.round(np.arange(0, 0.061, 0.01), 2)} for st_ in S}
        for T_ in (0.02, 0.04, 0.06):
            best = None
            n = int(round(T_ / 0.01))
            for a_ in itertools.product(range(n + 1), repeat=len(S)):
                if sum(a_) != n:
                    continue
                fs = [round(x * 0.01, 2) for x in a_]
                tot = mo + sum(Wf[st_][f] for st_, f in zip(S, fs))
                sd = tot[ok & dev]
                sh = sd.mean() / sd.std() * np.sqrt(52)
                if best is None or sh > best[0]:
                    best = (sh, fs)
            mixes[f"dev split of {T_ * 100:.0f}%: " + " ".join(f"{s_}{f * 100:.0f}" for s_, f in zip(S, best[1]))] = dict(zip(S, best[1]))
        W = {st_: {**W[st_], **Wf[st_]} for st_ in S}
        for lab, fs in mixes.items():
            tot = mo + sum(W[st_][f] for st_, f in fs.items())
            for era, msk in (("dev 2018-23", ok & dev), ("HOLDOUT 2024-26", ok & hold), ("all", ok)):
                res.append({**stats(tot[msk], name), "mix": lab, "era": era})
    R = pd.DataFrame(res)
    print("\nPICKED (dev): " + "; ".join(f"{k}: {dict(zip(key, v))}" for k, v in pick.items()))
    print("\nCORRELATIONS (weekly, sleeve at 2%/day)\n" + pd.DataFrame(corr).round(2).to_string(index=False))
    print("\nBOOK x SLEEVES\n" + R[["book", "mix", "era", "cagr_pct", "vol_pct", "sharpe", "max_dd_pct", "worst_week_pct"]]
          .round(2).to_string(index=False))
    R.to_csv(f"{A.RD}/results/mix_options_v2.csv", index=False)
    pd.DataFrame(corr).to_csv(f"{A.RD}/results/mix_options_v2_corr.csv", index=False)
    x = R[(R.book == "STACK_v2")]
    print("::notice::mix_options_v2 STACK_v2 (Sharpe / CAGR / maxDD; dev | HOLDOUT): " + "; ".join(
        f"{lab}: " + " | ".join(f"{r.sharpe:.2f} / {r.cagr_pct:+.0f}% / {r.max_dd_pct:.0f}%" for r in g[g.era != "all"].itertuples())
        for lab, g in x.groupby("mix", sort=False)) + " | picks: " + "; ".join(f"{k} {v[1]} k{v[2]} w{v[3]} {v[4]}" for k, v in pick.items()))
    c = pd.DataFrame(corr)
    c = c[c.book == "STACK_v2"]
    print("::notice::mix_options_v2 corr with STACK_v2 (dev/holdout) and sleeve in book's worst 10% weeks vs other (dev, %): " + "; ".join(
        f"{r.sleeve}: {r.corr_dev:+.2f}/{r.corr_hold:+.2f}, {r.sleeve_in_book_worst10pct_dev:+.2f} vs {r.sleeve_other_weeks_dev:+.2f}" for r in c.itertuples()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        print("::error::" + tb[-900:].replace("\n", " | "))
        raise
