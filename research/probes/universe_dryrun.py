"""
research/probes/universe_dryrun.py -- dry run of the fixed stage-1 universe (Oct 16, 2026).
Runs the SAME upsert path tonight's pipeline will run (update existing -> gate -> insert new)
on a copy of data/universe.csv, writes nothing back, and reports counts and spot checks
as ::notice:: lines. Never commits.
"""
import importlib.util, os, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yfinance>=0.2.40"], check=False)
import pandas as pd

sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("u", "pipeline/01_universe.py")
u = importlib.util.module_from_spec(spec); spec.loader.exec_module(u)
t0 = time.time()
old = pd.read_csv("data/universe.csv")
nasdaq, nyse, alpaca = u.fetch_nasdaq_ftp(), u.fetch_nyse_ftp(), u.fetch_alpaca_assets()
raw = u.combine_and_clean(nasdaq, nyse, alpaca)
print(f"::notice::sources: nasdaq {len(nasdaq):,}, nyse {len(nyse):,}, alpaca {len(alpaca):,} -> raw {len(raw):,}")
master = u.update_existing(old.copy())
master = u.check_gate_failures(master)
new = u.find_new_symbols(raw, master)
master = u.insert_new_symbols(new, raw, master)
TEST = ["MU", "PLTR", "UBER", "NOW", "INTU", "PANW", "MSTR", "COHR", "GLW", "TER", "MPWR", "SCHW", "MELI", "FICO",
        "SPCX", "CBRS", "HONA", "FDXF", "SNAP", "AMP", "WST", "CASY", "LOW", "KR", "PWR"]
have = set(master.symbol)
print(f"::notice::universe old {len(old):,} -> new {len(master):,} (+{len(set(master.symbol) - set(old.symbol)):,} added, "
      f"-{len(set(old.symbol) - set(master.symbol)):,} removed); {time.time() - t0:.0f}s")
print(f"::notice::spot check present: {[t for t in TEST if t in have]}")
print(f"::notice::spot check MISSING: {[t for t in TEST if t not in have]}")
miss = [t for t in TEST if t not in have]
for t in miss[:10]:
    r = raw[raw.symbol == t]
    print(f"::notice::  {t}: in raw list={len(r) > 0}")
ex = master.exchange.value_counts().to_dict()
print(f"::notice::exchange labels: {ex}")
master.to_csv("/tmp/universe_dryrun.csv", index=False)
