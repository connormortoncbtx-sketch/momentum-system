"""
research/extra_books.py -- full-period (2017-2026) weekly returns of two books that were
tested earlier but never stacked, using their PRE-REGISTERED designs (chosen on dev):
  PEAD     research/pead.py      sue_gap_up, events <= 13 weeks old, k=1, n=20, equal weight
           (dev Sharpe 0.77, holdout 0.78)
  INSIDER  research/insiders.py  rel_value + 12-1 momentum, 13-week window, k=4, n=30, equal
           (dev Sharpe 0.74, holdout 0.47)
Writes <PANEL>/results/extra_books.csv (formation-week index, like research/avoid_stack.py).
"""
import os, sys
import pandas as pd

RD = os.environ.get("PANEL", "research_data")
HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
for name, path, cut, call in (
        ("PEAD", "pead.py", "holdout_only =",
         "s = scores('sue_gap_up', 13); m = U.loc[s.index] & s.notna() & (s > 0); "
         "R = B.simulate(p, s, m, n=20, buffer=40, k=1, cost_bps=15, weighting='equal', vol=f['vol12'])"),
        ("INSIDER", "insiders.py", "grid = list(",
         "s = score('rel_value+m12_1', 13); "
         "R = B.simulate(p, s, U.loc[s.index] & s.notna(), n=30, buffer=60, k=4, cost_bps=15, weighting='equal', vol=f['vol12'])")):
    src = open(os.path.join(HERE, path)).read()
    src = src[: src.index(cut)]
    g = {"__name__": "extra", "__file__": os.path.join(HERE, path)}
    sys.argv = [path, RD]
    exec(compile(src + "\n" + call, path, "exec"), g)
    out[name] = g["R"]
    print(name, "weeks", len(g["R"]), flush=True)
df = pd.DataFrame(out)
os.makedirs(f"{RD}/results", exist_ok=True)
df.to_csv(f"{RD}/results/extra_books.csv")
print(df.describe().round(4).to_string())
