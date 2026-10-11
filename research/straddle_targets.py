"""
research/straddle_targets.py -- choose (week, symbol) pairs for the options test.

Train the spike and crash models of research/signatures_model.py on 2017-2023 only,
score Feb 2024 - Sep 2026 (all out-of-sample), and keep, each week, among names
trading >= $5M/day (so listed options are likely):
  'model'   : the 25 names with the highest P(spike) + P(crash)  (a big move predicted)
  'control' : 25 random other names from the same liquid set
Output: research/samples/straddle_targets.parquet
"""
import os
import numpy as np, pandas as pd
import lightgbm as lgb

CACHE = os.environ.get("CACHE", "research/results/signatures_table.parquet")
df = pd.read_parquet(CACHE)
TARGETS = ["spike1", "crash1", "run13", "next1_ex", "next13", "next1_monclose"]
feats = [c for c in df.columns if c not in ["week", "symbol"] + TARGETS]
dev = df[df.week < "2024-01-01"]
te = df[(df.week >= "2024-01-26") & (np.exp(df.log_adv) >= 5e6)].copy()
for tgt in ("spike1", "crash1"):
    tr = dev.dropna(subset=[tgt])
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=1)
    m.fit(tr[feats], tr[tgt])
    te[f"p_{tgt}"] = m.predict_proba(te[feats])[:, 1]
te["p_move"] = te.p_spike1 + te.p_crash1
rng = np.random.default_rng(11)
out = []
for wk, g in te.groupby("week"):
    top = g.nlargest(25, "p_move").assign(group="model")
    rest = g.drop(top.index)
    ctl = rest.iloc[rng.permutation(len(rest))[:25]].assign(group="control")
    out += [top, ctl]
o = pd.concat(out)[["week", "symbol", "group", "p_spike1", "p_crash1", "p_move", "vol12", "log_price"]]
os.makedirs("research/samples", exist_ok=True)
o.to_parquet("research/samples/straddle_targets.parquet", index=False)
print(o.groupby("group").size(), o.week.nunique(), "weeks;", o.week.min().date(), "->", o.week.max().date())
print(o.groupby("group")[["p_move", "vol12"]].median())
