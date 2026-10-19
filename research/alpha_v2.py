"""
research/alpha_v2.py -- rebuild "alpha" and "EV" the way the research says they should be built (Oct 19, 2026).

The live alpha (LightGBM on short-term RS / volume / sentiment, weekly target, retrained on recent months) and EV
showed no signal on the 23 logged live weeks. Two replacements, both from price/volume features a weekly shadow can
compute from Alpaca bars (research/backtest.py features + r13w):

  BLEND (v2b)  the 6 features with the strongest dev (2017-2022) 4-week rank IC, each cross-sectionally ranked,
               signed by its dev IC, weighted by |IC|
  ML    (v2a)  LightGBM regression on the weekly cross-sectional rank of the NEXT 4-WEEK excess return; inputs are the
               features' cross-sectional ranks; trained 2017-2022, early-stopped on 2023, scored once on 2024-2026
  ALPHA prob   isotonic map (fit on 2023) from score to P(next 4-week excess > 0) -- a calibrated probability
  EV           score decile -> mean next 4-week excess (fit on 2017-2023); tested as a filter (EV > 0)
Benchmarks: the LIVE proxy composite, plain 12-1 momentum (the SLOW book's signal).
Books: weekly, top 20 by score, each held 4 weeks (4 overlapping tranches), Friday close to Friday close,
15 bp per side on the replaced tranche; excess vs the equal-weight universe.
Universe: research default (price >= $5, >= $2M/day, >= 53 weeks history, no funds/splits flags).
Artifacts for the shadow: results/alpha_v2_blend.json, results/alpha_v2_lgbm.txt, results/alpha_v2_calib.json
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

RD = os.environ.get("RESEARCH_DATA", os.environ.get("PANEL", "research_data"))
os.environ.setdefault("PANEL", RD)
OUT = f"{RD}/results"; os.makedirs(OUT, exist_ok=True)
DEV_END, VAL_END = pd.Timestamp("2022-12-31"), pd.Timestamp("2023-12-31")
H, N, COST = 4, 20, 0.0015
FEATS = ["r1w", "r4w", "r13w", "m12_1", "m6_1", "m3_1", "hi52", "vol12", "trend", "ladv", "lprice"]


def build():
    p = B.load_panel(RD, "2026-10-02")
    f = B.features(p)
    U = B.universe(p, f, 5.0, 2e6)
    C = p.df("close")
    f["r13w"] = C / C.shift(13) - 1
    f["ladv"] = np.log(f["adv"].clip(lower=1))
    f["lprice"] = np.log(f["price"].clip(lower=0.01))
    nxt1 = (C.shift(-1) / C - 1).clip(-0.9, 3)
    ew1 = nxt1.where(U).mean(axis=1)
    ex1 = nxt1.sub(ew1, axis=0)
    fwd4 = (C.shift(-H) / C - 1).clip(-0.95, 5)
    ex4 = fwd4.sub(fwd4.where(U).mean(axis=1), axis=0)
    R = {k: B.xrank(f[k], U) for k in FEATS}
    live = (0.35 * B.xrank(f["rs_live"], U) + 0.25 * B.xrank(f["trend"], U) + 0.20 * B.xrank(f["hi52"], U)) / 0.80
    return p, U, R, live, ex1, ex4


def long(R, U, ex4):
    d = {k: v.where(U).stack(future_stack=True) for k, v in R.items()}
    df = pd.DataFrame(d)
    df["y"] = ex4.where(U).stack(future_stack=True).reindex(df.index)
    df = df.dropna(subset=FEATS, how="all")
    df.index.names = ["week", "symbol"]
    return df.reset_index()


def ic_table(df):
    rows = []
    dev = df[df.week <= DEV_END].dropna(subset=["y"])
    for k in FEATS:
        ic = dev.groupby("week").apply(lambda g: g[k].corr(g.y, method="spearman"))
        rows.append(dict(feature=k, ic=ic.mean(), t=ic.mean() / ic.std() * np.sqrt(len(ic) / H)))   # /H: overlapping 4w targets
    return pd.DataFrame(rows).sort_values("t", key=abs, ascending=False)


def book(score, U, ex1, label):
    """Top-N each week, held H weeks as overlapping tranches; weekly excess return net of costs."""
    s = score.where(U)
    picks = {w: list(s.loc[w].dropna().nlargest(N).index) for w in s.index if s.loc[w].notna().sum() >= 3 * N}
    weeks = list(s.index)
    out = {}
    for i, w in enumerate(weeks):
        trs = [picks.get(weeks[i - k]) for k in range(H) if i - k >= 0 and weeks[i - k] in picks]
        if len(trs) < H:
            continue
        r = np.mean([ex1.loc[w, t].mean() for t in trs]) - 2 * COST / H
        out[w] = r
    r = pd.Series(out).dropna()
    rows = []
    for per, m in (("dev", r.index <= VAL_END), ("hold", r.index > VAL_END)):
        x = r[m]
        if len(x) < 30:
            continue
        rows.append(dict(book=label, period=per, weeks=len(x), excess_pct_wk=x.mean() * 100,
                         excess_ann_pct=((1 + x).prod() ** (52 / len(x)) - 1) * 100, sharpe=x.mean() / x.std() * np.sqrt(52),
                         hit_pct=(x > 0).mean() * 100))
    return rows, r


def main():
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    p, U, R, live, ex1, ex4 = build()
    df = long(R, U, ex4)
    print(f"rows {len(df):,}, weeks {df.week.nunique()}", flush=True)
    pd.set_option("display.width", 220)

    # ---- v2b: dev-IC weighted blend
    ict = ic_table(df)
    print("DEV 4-WEEK RANK IC by feature (t adjusted for overlap)\n" + ict.round(4).to_string(index=False))
    top = ict.head(6)
    wts = {r.feature: float(np.sign(r.ic) * abs(r.ic)) for r in top.itertuples()}
    tot = sum(abs(v) for v in wts.values())
    wts = {k: v / tot for k, v in wts.items()}
    print("BLEND weights:", {k: round(v, 3) for k, v in wts.items()})
    blend = sum(abs(w) * (R[k] if w > 0 else 1 - R[k]) for k, w in wts.items())
    json.dump({"features": wts, "horizon_weeks": H, "n": N}, open(f"{OUT}/alpha_v2_blend.json", "w"), indent=1)

    # ---- v2a: LightGBM on per-week rank of the 4-week excess return
    df["yr"] = df.groupby("week").y.rank(pct=True)
    tr, va = df[(df.week <= DEV_END) & df.yr.notna()], df[(df.week > DEV_END) & (df.week <= VAL_END) & df.yr.notna()]
    # embargo: drop the last H weeks of train so its labels don't overlap validation
    tr = tr[tr.week <= DEV_END - pd.Timedelta(weeks=H)]
    m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=31, min_child_samples=500, subsample=0.7,
                          subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)
    m.fit(tr[FEATS], tr.yr, eval_set=[(va[FEATS], va.yr)], callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"LightGBM best iteration {m.best_iteration_}; gain importance:",
          dict(sorted(zip(FEATS, m.booster_.feature_importance("gain").round(0)), key=lambda x: -x[1])))
    m.booster_.save_model(f"{OUT}/alpha_v2_lgbm.txt")
    df["ml"] = m.predict(df[FEATS])
    ml = df.pivot(index="week", columns="symbol", values="ml").reindex(index=R["m12_1"].index, columns=R["m12_1"].columns)
    ml_r = B.xrank(ml, U)

    # ---- calibration (alpha prob) and EV, fit on 2023 / dev; checked on holdout
    va2 = df[(df.week > DEV_END) & (df.week <= VAL_END)].dropna(subset=["y"])
    iso = IsotonicRegression(out_of_bounds="clip").fit(va2.ml, (va2.y > 0).astype(float))
    df["alpha_prob"] = iso.predict(df.ml)
    df["dec"] = df.groupby("week").ml.rank(pct=True).mul(10).clip(upper=9.999).astype(int)
    ev_map = df[(df.week <= VAL_END)].dropna(subset=["y"]).groupby("dec").y.mean()
    df["ev"] = df.dec.map(ev_map)
    xs = np.linspace(df.ml.min(), df.ml.max(), 50)
    json.dump({"iso_x": xs.tolist(), "iso_y": iso.predict(xs).tolist(), "ev_by_decile": ev_map.round(5).tolist()},
              open(f"{OUT}/alpha_v2_calib.json", "w"))
    h = df[df.week > VAL_END].dropna(subset=["y"])
    h["pb"] = pd.cut(h.alpha_prob, [0, .4, .45, .5, .55, .6, 1])
    print("\nCALIBRATION on holdout: alpha_prob bin -> realised share beating the universe over 4 weeks")
    print(h.groupby("pb", observed=True).agg(n=("y", "size"), mean_prob=("alpha_prob", "mean"),
                                              realised=("y", lambda v: (v > 0).mean())).round(3).to_string())
    print("EV by decile (dev fit, % per 4 wk):", (ev_map * 100).round(2).tolist())
    print("EV decile -> holdout realised (% per 4 wk):", (h.groupby("dec").y.mean() * 100).round(2).tolist())

    # ---- books
    evf = df.pivot(index="week", columns="symbol", values="ev").reindex(index=ml.index, columns=ml.columns)
    rows, series = [], {}
    for label, sc in (("LIVE proxy", live), ("12-1 momentum", R["m12_1"]), ("v2b BLEND", blend), ("v2a ML", ml_r),
                      ("v2a ML + EV>0 filter", ml_r.where(evf > 0))):
        r_, s_ = book(sc, U, ex1, label)
        rows += r_; series[label] = s_
    Rb = pd.DataFrame(rows)
    Rb.to_csv(f"{OUT}/alpha_v2_books.csv", index=False)
    W = Rb.pivot(index="book", columns="period", values=["excess_pct_wk", "excess_ann_pct", "sharpe", "hit_pct"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    print("\nBOOKS: top 20, 4-week hold, excess vs equal-weight universe after 15 bp/side\n" + W.round(3).to_string())
    pd.DataFrame(series).to_csv(f"{OUT}/alpha_v2_weekly.csv")
    for b in ("v2b BLEND", "v2a ML", "12-1 momentum", "LIVE proxy"):
        x = Rb[(Rb.book == b)].set_index("period")
        print(f"::notice::alpha v2 {b}: excess {x.at['dev','excess_ann_pct']:+.1f}%/yr Sharpe {x.at['dev','sharpe']:.2f} (dev 2017-23) -> "
              f"{x.at['hold','excess_ann_pct']:+.1f}%/yr Sharpe {x.at['hold','sharpe']:.2f} (holdout 2024-26)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
