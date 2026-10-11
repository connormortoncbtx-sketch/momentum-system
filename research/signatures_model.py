"""research/signatures_model.py -- lift tables, LightGBM models and payoff for the
pre-event signature study. Input: the table written by research/signatures.py."""
import os
import numpy as np, pandas as pd
OUT = os.environ.get("OUT", "research/results")
CACHE = os.environ.get("CACHE", f"{OUT}/signatures_table.parquet")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
df = pd.read_parquet(CACHE)
TARGETS = ["spike1", "crash1", "run13", "next1_ex", "next13", "next1_monclose"]
feats = [c for c in df.columns if c not in ["week", "symbol"] + TARGETS]
print(f"rows {len(df):,}, weeks {df.week.nunique()}, features {len(feats)}")
dev = df[df.week < "2024-01-01"]
hold = df[df.week >= "2024-01-01"]
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)

# 1. single-feature lift: base rate vs rate in the top decile of each feature (dev | holdout)
rows = []
for tgt in ():   # lift table already produced (see signatures_lift.csv); skip to save time
  pass
for tgt in ("spike1", "run13"):
    for c in feats:
        for nm, d in (("dev", dev), ("hold", hold)):
            x = d[[c, tgt]].dropna()
            if len(x) < 1000 or x[c].nunique() < 3:
                continue
            q = x[c].rank(pct=True)
            base = x[tgt].mean()
            rows.append({"target": tgt, "feature": c, "period": nm, "base%": base * 100,
                         "top10%_rate%": x[tgt][q > 0.9].mean() * 100,
                         "bottom10%_rate%": x[tgt][q <= 0.1].mean() * 100})
lift = pd.DataFrame(rows)
lift["lift_top"] = lift["top10%_rate%"] / lift["base%"]
lt = lift.pivot_table(index=["target", "feature"], columns="period", values="lift_top").round(2)
print("\nLift: event rate in the top 10% of each feature / base rate")
print(lt.sort_values(("dev"), ascending=False).to_string())

# 2. models
res = {}
for tgt in ("spike1", "crash1", "run13"):
    tr = dev.dropna(subset=[tgt])
    if tgt == "run13":                         # avoid overlap leakage into the holdout
        tr = tr[tr.week < pd.Timestamp("2024-01-01") - pd.Timedelta(weeks=13)]
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(tr[feats], tr[tgt])
    te = hold.dropna(subset=[tgt]).copy()
    te["p"] = m.predict_proba(te[feats])[:, 1]
    auc = roc_auc_score(te[tgt], te.p)
    base = te[tgt].mean()
    top = te[te.p.rank(pct=True) > 0.99]
    res[tgt] = (m, te)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print(f"\n### {tgt}: holdout AUC {auc:.3f}; base rate {base * 100:.2f}%; top-1% predicted rate "
          f"{top[tgt].mean() * 100:.2f}% (lift {top[tgt].mean() / base:.1f}x)")
    print("    top features by gain: " + ", ".join(f"{k} {v / imp.sum():.0%}" for k, v in imp.head(10).items()))

# 3. payoff: each holdout week buy the 10 highest P(spike) - P(crash), hold 1 week; and top-20 P(run13), hold 13 weeks
sp, cr = res["spike1"][1], res["crash1"][1]
both = sp.merge(cr[["week", "symbol", "p"]], on=["week", "symbol"], suffixes=("_s", "_c"))
both["net"] = both.p_s - both.p_c
for score in ("p_s", "net"):
    picks = both.sort_values(score, ascending=False).groupby("week").head(10)
    wk = picks.groupby("week").agg(ex=("next1_ex", "mean"), monc=("next1_monclose", "mean"),
                                   spikes=("spike1", "sum"), crashes=("crash1", "mean"))
    print(f"\nPayoff, top 10/week by {score}: mean next-week excess {wk.ex.mean() * 100:+.2f}% "
          f"(Mon-close entry {wk.monc.mean() * 100:+.2f}%), median {wk.ex.median() * 100:+.2f}%, "
          f"spikes caught/week {wk.spikes.mean():.2f}, t {wk.ex.mean() / (wk.ex.std() / np.sqrt(len(wk))):.2f}")
rn = res["run13"][1]
picks = rn.sort_values("p", ascending=False).groupby("week").head(20)
w13 = picks.groupby("week").agg(r=("next13", "mean"), hit=("run13", "mean"))
allu = rn.groupby("week").next13.mean()
print(f"\nPayoff, top 20/week by P(run13), held 13 weeks: mean {w13.r.mean() * 100:+.1f}% vs universe "
      f"{allu.mean() * 100:+.1f}%; hit rate {w13.hit.mean() * 100:.1f}% vs base {rn.run13.mean() * 100:.1f}%")
os.makedirs(OUT, exist_ok=True)
lift.to_csv(f"{OUT}/signatures_lift.csv", index=False)

# 4. closer look at the run13 picks (holdout): distribution and non-overlapping blocks
x = picks.copy()
print("\nrun13 picks: next-13-week return distribution (holdout):",
      {k: round(v * 100, 1) for k, v in x.next13.describe(percentiles=[.1, .25, .5, .75, .9]).items() if k != "count"})
print(f"  share losing > 30%: {(x.next13 < -0.3).mean() * 100:.1f}% (universe {(rn.next13 < -0.3).mean() * 100:.1f}%)")
ex13 = w13.r - allu.reindex(w13.index)
for off in range(0, 13, 4):
    blk = ex13.iloc[off::13]
    print(f"  non-overlapping start offset {off}: {len(blk)} blocks, mean excess {blk.mean() * 100:+.1f}%, "
          f"t {blk.mean() / (blk.std() / np.sqrt(len(blk))):.2f}")
# dev check of the same rule (in-sample for the model, so optimistic)
m_run = res["run13"][0]
dv = dev.dropna(subset=["run13"]).copy()
dv["p"] = m_run.predict_proba(dv[feats])[:, 1]
pk = dv.sort_values("p", ascending=False).groupby("week").head(20)
print(f"  (in-sample dev, optimistic) top-20 13wk mean {pk.groupby('week').next13.mean().mean() * 100:+.1f}% "
      f"vs universe {dv.groupby('week').next13.mean().mean() * 100:+.1f}%")
# simple volatility-only benchmark: top 20 by vol12 each week
vb = rn.sort_values("vol12", ascending=False).groupby("week").head(20).groupby("week").next13.mean()
print(f"  benchmark, top-20 by 12-week volatility alone: {vb.mean() * 100:+.1f}%")
