"""
research/llm_sample.py -- choose the small-cap 8-K sample for the LLM filing study.

Run locally with the research-data checkout (PANEL) and the sec/ files (SEC).
Writes research/samples/llm_8k_sample.parquet: accession, cik, symbol, items,
accepted_et (stored; see note), market cap and trailing revenue known before the
filing, period (dev / holdout).

Timing note (verified Oct 2026 with research/probes/sec_time*.py): the SEC
submissions API's acceptanceDateTime is sometimes true UTC and sometimes UTC
plus one extra ET offset, so the stored ET value is either right or 4-5h LATE.
It is never early, so using it is conservative. The filing study re-reads the
authoritative 'Accepted' time from each filing's index page.
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

PANEL = os.environ.get("PANEL", "research_data")
SEC = os.environ.get("SEC", f"{PANEL}/sec")
ITEMS = {"1.01", "1.02", "1.03", "2.01", "2.03", "2.04", "2.05", "2.06", "3.01", "3.02",
         "4.01", "4.02", "5.01", "5.02", "7.01", "8.01"}
DEV0, HOLD0, END = "2023-01-01", "2025-04-01", "2026-09-25"   # holdout starts after the model's training cutoff
N_DEV, N_HOLD = 5000, 12000
# FULL=1: every qualifying 8-K from FULL_START on that is NOT already in the study sample
# (full-coverage backfill for portfolio tests) -> research/samples/llm_8k_full.parquet
FULL = os.environ.get("FULL") == "1"
FULL_START = os.environ.get("FULL_START") or "2024-01-01"
FULL_END = os.environ.get("FULL_END") or END
FULL_OUT = os.environ.get("FULL_OUT") or "research/samples/llm_8k_full.parquet"

_files = sorted(glob.glob(f"{SEC}/filings_20[12][0-9].parquet"))
if os.environ.get("DELISTED_ONLY") == "1":          # audit redo: filers that later delisted
    _files = [f"{SEC}/filings_delisted.parquet"]
fi = pd.concat(pd.read_parquet(f) for f in _files)
k = fi[(fi.form == "8-K") & (fi.accepted_et >= (FULL_START if FULL else DEV0)) & (fi.accepted_et < (FULL_END if FULL else END))].copy()
k["item_set"] = k["items"].fillna("").str.split(",").apply(lambda xs: {x.strip() for x in xs})
k = k[k.item_set.apply(lambda s: bool(s & ITEMS))]
k["symbol"] = k.tickers.str.split(",").str[0]
print("8-Ks with target items:", len(k))

p = B.load_panel(PANEL, "2026-10-02"); f = B.features(p)
fri = k.accepted_et.dt.normalize() + pd.to_timedelta((4 - k.accepted_et.dt.weekday) % 7, unit="D") - pd.Timedelta(days=7)
i = p.weeks.get_indexer(fri); j = p.symbols.get_indexer(k.symbol)
ok = (i >= 0) & (j >= 0)
k = k[ok].copy(); i, j = i[ok], j[ok]
k["price"] = f["price"].to_numpy()[i, j]
k["adv"] = f["adv"].to_numpy()[i, j]

fu = pd.concat(pd.read_parquet(f, columns=["cik", "tag", "qtrs", "value", "accepted_et", "filed"])
               for f in sorted(glob.glob(f"{SEC}/fundamentals_20*.parquet")))
fu = fu.dropna(subset=["accepted_et"]).sort_values("accepted_et")
sh = fu[fu.tag.isin(["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"])][["cik", "accepted_et", "value"]]
rev = fu[fu.tag.isin(["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]) & (fu.qtrs == 4)][["cik", "accepted_et", "value"]]
k = k.sort_values("accepted_et")
k = pd.merge_asof(k, sh.rename(columns={"value": "shares", "accepted_et": "sh_t"}), left_on="accepted_et",
                  right_on="sh_t", by="cik", direction="backward")
k = pd.merge_asof(k, rev.rename(columns={"value": "revenue_ttm", "accepted_et": "rev_t"}), left_on="accepted_et",
                  right_on="rev_t", by="cik", direction="backward")
k["mcap"] = k.shares * k.price
k = k[(k.mcap >= 30e6) & (k.mcap <= 2e9) & (k.price >= 1) & (k.adv >= 3e5)]
k["period"] = np.where(k.accepted_et >= HOLD0, "holdout", "dev")
print(k.groupby("period").size())
rng = np.random.default_rng(7)
if FULL:
    done = set()
    for _s in glob.glob("research/samples/llm_8k_*.parquet"):
        done |= set(pd.read_parquet(_s, columns=["accession"]).accession)
    sel = k[(k.accepted_et >= FULL_START) & (k.accepted_et < FULL_END) & ~k.accession.isin(done)]
else:
    dev = k[k.period == "dev"]; hold = k[k.period == "holdout"]
    dev = dev.iloc[rng.permutation(len(dev))[:N_DEV]]
    hold = hold.iloc[rng.permutation(len(hold))[:N_HOLD]]
    sel = pd.concat([dev, hold])
out = sel[["accession", "cik", "symbol", "items", "accepted_et", "filing_date", "primary_doc",
                              "sic", "price", "adv", "mcap", "revenue_ttm", "period"]].sort_values("accepted_et")
os.makedirs("research/samples", exist_ok=True)
out.to_parquet(FULL_OUT if FULL else "research/samples/llm_8k_sample.parquet", index=False)
print(out.groupby("period").size(), out.mcap.describe())
items = out["items"].str.split(",").explode().str.strip()
print(items[items.isin(ITEMS)].value_counts())
