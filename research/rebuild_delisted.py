"""
research/rebuild_delisted.py -- merge the EODHD-listed delisted stocks into the weekly research panel (Nov 14, 2026):
adjusted weekly rows (fetch_history.py --status eodhd), then their unadjusted closes (--raw --status eodhd), so the
2023-26 period stops being survivor-only. Runs in Actions via research_run.yml.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_history as FH  # noqa: E402

RD = os.environ.get("RESEARCH_DATA", "research_data")
for extra in ([], ["--raw"]):
    sys.argv = ["fetch_history.py", "--out", RD, "--status", "eodhd"] + extra
    FH.main()
import glob
import pandas as pd
L = pd.concat([pd.read_parquet(f, columns=["week", "symbol"]) for f in glob.glob(f"{RD}/weekly_*.parquet")])
last = L.groupby("symbol").week.max()
gone = last[last < last.max() - pd.Timedelta(weeks=4)]
by = gone.groupby(gone.dt.year).size()
print("::notice::panel after delisted merge: symbols " + f"{L.symbol.nunique():,}; stopped trading by year " + str(by.to_dict()))
