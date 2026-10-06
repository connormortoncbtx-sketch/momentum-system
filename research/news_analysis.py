"""
research/news_analysis.py -- do Claude's news labels predict post-event returns
beyond the price reaction? Hypotheses fixed before labels were seen:

  H1  fundamental_signal coefficient on abnormal return, controlling for r2_abn
  H2  'underreacted' AND fundamental_signal >= 2  vs all other labelled events
  H3  guidance raised vs lowered

Horizons: 8 weeks for dev/holdout; 4 weeks for post_cutoff (data ends Oct 2026).
post_cutoff (Jul-Sep 2026, after the model's training cutoff) is the only split
immune to the model remembering outcomes.

    python research/news_analysis.py <research-data dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = sys.argv[1]
OUT = Path("research/results")
lab = pd.read_parquet(f"{DATA}/news_labels.parquet")
es = pd.read_parquet(OUT / "pead_event_study.parquet")
lab["week"] = pd.to_datetime(lab["week"])
df = lab.merge(es[["symbol", "week", "r2_abn", "sue", "abn_1w", "abn_4w", "abn_8w", "abn_12w"]],
               on=["symbol", "week"], how="left")
print("label status by split:")
print(pd.crosstab(df.split, df.status))
ok = df[(df.status == "ok") & (df.news_relevant == True)].copy()  # noqa: E712
print(f"\nusable labelled events: {len(ok)}")


def ols(y, X):
    X = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    s2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, b / se


rows = []
for split, h in [("dev", "abn_8w"), ("holdout", "abn_8w"), ("post_cutoff", "abn_4w")]:
    d = ok[ok.split == split].dropna(subset=[h, "r2_abn", "fundamental_signal"])
    if len(d) < 20:
        print(f"{split}: only {len(d)} events with returns, skipping")
        continue
    y = d[h].clip(-0.5, 1.0).values
    b, t = ols(y, d[["fundamental_signal", "r2_abn"]].values)
    strong = (d.reaction_assessment == "underreacted") & (d.fundamental_signal >= 2)
    up, dn = d[d.guidance == "raised"][h], d[d.guidance == "lowered"][h]
    sig_mean = d.groupby("fundamental_signal")[h].mean() * 100
    rows.append({
        "split": split, "horizon": h, "n": len(d),
        "H1 coef per signal pt %": b[1] * 100, "H1 t": t[1],
        "H2 strong n": int(strong.sum()), "H2 strong mean %": d[strong][h].mean() * 100,
        "H2 rest mean %": d[~strong][h].mean() * 100,
        "H3 raised %": up.mean() * 100, "H3 lowered %": dn.mean() * 100,
        "H3 n raised/lowered": f"{len(up)}/{len(dn)}",
    })
    print(f"\n{split} ({h}) mean abnormal return % by fundamental_signal:")
    print(pd.DataFrame({"mean%": sig_mean.round(2), "n": d.groupby("fundamental_signal").size()}).T.to_string())

res = pd.DataFrame(rows).set_index("split")
print("\n" + res.round(2).T.to_string())
res.to_csv(OUT / "news_hypotheses.csv")
print(f"\nspend: ${df.spent_usd.max():.2f}" if "spent_usd" in df else "")
