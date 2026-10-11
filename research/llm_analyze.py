"""research/llm_analyze.py -- score the LLM filing-reader study (dev designs, holdout once)."""
import os, sys
import numpy as np, pandas as pd
D = os.environ.get("LLM", "research_data/llm")
ev = pd.read_parquet(f"{D}/events.parquet").merge(pd.read_parquet(f"{D}/judgements.parquet"), on="accession")
ev = ev[ev.ok == True].copy()
ev["n_flags"] = ev.red_flags.fillna("").apply(lambda s: len([x for x in s.split("|") if x]))
ev["dm"] = ev.direction * ev.materiality
ev["dmn"] = ev.direction * ev.materiality * ev.novelty
ev["dval_mcap"] = ev.dollar_value_usd / ev.mcap
H = ["ab_oc0", "ab_1", "ab_5", "ab_20"]
for c in H + ["gap"]:
    ev[c] = ev[c].clip(-0.9, 2.0)          # cap absurd prints
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)


def tstat(x):
    x = x.dropna(); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 20 else np.nan


def table(d, by):
    g = d.groupby(by, observed=True)
    out = g[H].mean() * 100
    out.insert(0, "n", g.size())
    out["t_ab_5"] = g.ab_5.apply(tstat)
    out["gap%"] = g.gap.mean() * 100
    return out.round(2)


part = os.environ.get("PART", "dev")
d = ev[ev.period == part]
print(f"=== {part}: {len(d):,} filings; IWM-adjusted returns from the FIRST OPEN after acceptance (%)")
print("\nBy LLM direction:"); print(table(d, "direction"))
d = d.assign(dm_bin=pd.cut(d.dm, [-2.01, -0.75, -0.25, -0.01, 0.01, 0.25, 0.75, 2.0]))
print("\nBy direction x materiality:"); print(table(d, "dm_bin"))
print("\nBy event type (n >= 40):")
t = table(d, "event_type"); print(t[t.n >= 40].sort_values("ab_5"))
d = d.assign(flags=pd.cut(d.n_flags, [-1, 0, 1, 2, 3, 99], labels=["0", "1", "2", "3", "4+"]))
print("\nBy number of red flags:"); print(table(d, "flags"))
pos = d[(d.direction > 0)]
pos = pos.assign(size=pd.cut(pos.dval_mcap, [0, 0.05, 0.2, 1, 1e9], labels=["<5% of mcap", "5-20%", "20-100%", ">100%"]))
print("\nPositive-direction filings by stated $ value relative to market cap:"); print(table(pos, "size"))
# correlation of score with returns (rank IC)
for s in ("direction", "dm", "dmn"):
    print(f"rank corr {s:9s} vs " + "  ".join(f"{h}: {d[s].corr(d[h], method='spearman'):+.3f}" for h in H))
cost_in, cost_out = ev.in_tokens.sum(), ev.out_tokens.sum()
print(f"\ntokens: in {cost_in:,} out {cost_out:,} -> batch cost ~${cost_in / 1e6 * 0.5 + cost_out / 1e6 * 2.5:,.2f} "
      f"(${(cost_in / 1e6 * 0.5 + cost_out / 1e6 * 2.5) / len(ev) * 1000:.2f} per 1,000 filings)")
