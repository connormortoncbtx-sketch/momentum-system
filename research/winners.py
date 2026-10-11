"""
research/winners.py -- winners first: find the biggest winners, then dissect them.

A. Who were they BEFORE the run? Top 1% by next-13-week return (quarterly starts),
   2017-2026. For each feature in the signature table (all known at the start
   week) the winners' average percentile vs the universe (50 = typical), next to
   the biggest LOSERS (bottom 1%). Traits that winners and losers share
   (volatility) say "a big move", not "a win"; the useful ones differ between them.
B. Where were they? SIC industry concentration of winners vs the universe, by year.
C. Is the clustering tradeable? Industry momentum: rank stocks by their industry's
   equal-weight return over the last 13 weeks (SIC 3-digit, >= 5 members) and
   measure the next 13 weeks. Dev 2017-2023, holdout 2024-2026 (Newey-West t).
"""
import glob, os
import numpy as np, pandas as pd

P = os.environ.get("PANEL", "research_data")
CACHE = os.environ.get("CACHE", "research/results/signatures_table.parquet")
df = pd.read_parquet(CACHE)
T = ["spike1", "crash1", "run13", "next1_ex", "next13", "next1_monclose"]
feats = [c for c in df.columns if c not in ["week", "symbol"] + T]
weeks = sorted(df.week.unique())
qstarts = set(weeks[::13])
q = df[df.week.isin(qstarts) & df.next13.notna()].copy()
for c in feats:
    q[c] = q.groupby("week")[c].rank(pct=True) * 100
q["r_pct"] = q.groupby("week").next13.rank(pct=True)
win, lose = q[q.r_pct > 0.99], q[q.r_pct <= 0.01]
mid = q[(q.r_pct > 0.25) & (q.r_pct <= 0.75)]
prof = pd.DataFrame({"winners": win[feats].mean(), "losers": lose[feats].mean(), "middle_half": mid[feats].mean()})
prof["win_minus_lose"] = prof.winners - prof.losers
pd.set_option("display.width", 200)
print(f"A. {len(win):,} quarterly top-1% winners (median next-13wk {win.next13.median() * 100:.0f}%), "
      f"{len(lose):,} bottom-1% losers (median {lose.next13.median() * 100:.0f}%)")
print("   average percentile at the START of the quarter (50 = typical stock)")
print(prof.round(1).sort_values("win_minus_lose").to_string())

# B. industries
fi = pd.concat(pd.read_parquet(x, columns=["cik", "tickers", "sic"]) for x in sorted(glob.glob(f"{P}/sec/filings_*.parquet")))
fi = fi.drop_duplicates("cik")
fi = fi.assign(symbol=fi.tickers.str.split(",")).explode("symbol")
sic = fi[fi.sic.astype(str).str.len() >= 3].drop_duplicates("symbol").set_index("symbol").sic.astype(str)
q["sic2"] = q.symbol.map(sic.str[:2]); q["sic3"] = q.symbol.map(sic.str[:3])
q["year"] = pd.to_datetime(q.week).dt.year
NAMES = {"28": "chemicals/pharma", "73": "software/services", "36": "electronics/semis", "38": "instruments/medtech",
         "35": "machinery/computers", "60": "banks", "67": "holding/investment cos", "49": "utilities/energy svc",
         "13": "oil & gas", "62": "brokers/crypto-ish", "48": "telecom", "37": "transport equip", "80": "health services",
         "10": "metal mining", "33": "primary metals", "50": "wholesale", "59": "retail", "45": "air transport",
         "61": "credit/lending", "63": "insurance", "87": "engineering/research svcs", "29": "petroleum refining",
         "12": "coal", "16": "heavy construction", "15": "building construction", "34": "fabricated metals"}
w = q[q.r_pct > 0.99]
share = (w.sic2.value_counts(normalize=True) / q.sic2.value_counts(normalize=True)).dropna()
top_ind = w.sic2.value_counts().head(10)
print("\nB. industries of winners (SIC 2-digit): count, share of winners, over-representation vs universe")
print(pd.DataFrame({"name": [NAMES.get(i, "") for i in top_ind.index], "winners": top_ind.values,
                    "share%": (top_ind / len(w) * 100).round(1).values,
                    "x_universe": share.reindex(top_ind.index).round(2).values}, index=top_ind.index).to_string())
by_year = w.groupby("year").sic2.agg(lambda s: ", ".join(f"{k} {NAMES.get(k, '')} ({v})" for k, v in s.value_counts().head(3).items()))
print("\n   top 3 industries among winners, by year:"); print(by_year.to_string())
conc = w.groupby("year").sic3.agg(lambda s: s.value_counts(normalize=True).head(5).sum() * 100)
base = q.groupby("year").sic3.agg(lambda s: s.value_counts(normalize=True).head(5).sum() * 100)
print("\n   share of winners in the year's top-5 SIC-3 industries vs universe's top-5 share:")
print(pd.DataFrame({"winners_top5%": conc.round(0), "universe_top5%": base.round(0)}).to_string())

# C. industry momentum (weekly, all weeks)
a = df[["week", "symbol", "next13"]].copy()
a["sic3"] = a.symbol.map(sic.str[:3])
past = df[["week", "symbol", "r13w", "m12_1"]].copy()
a = a.merge(past, on=["week", "symbol"])
grp = a.groupby(["week", "sic3"])
a["ind_n"] = grp.symbol.transform("size")
a["ind_r13"] = (grp.r13w.transform("sum") - a.r13w.fillna(0)) / (a.ind_n - 1)      # leave-one-out
a = a[(a.ind_n >= 5) & a.next13.notna() & a.ind_r13.notna()]
a["ex13"] = a.next13 - a.groupby("week").next13.transform("mean")


def nw_t(x, lag=12):
    x = x.dropna().to_numpy(); n = len(x); e = x - x.mean(); v = e @ e / n
    for l in range(1, lag + 1):
        v += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(v / n)


rows = []
for name, col in (("industry 13wk momentum", "ind_r13"), ("own 13wk momentum", "r13w"), ("own 12-1 momentum", "m12_1")):
    b = a.dropna(subset=[col]).copy()
    b["qq"] = b.groupby("week")[col].rank(pct=True)
    top = b[b.qq > 0.8].groupby("week").ex13.mean(); bot = b[b.qq <= 0.2].groupby("week").ex13.mean()
    sp = top - bot
    for per, sl in (("dev", sp.index < pd.Timestamp("2024-01-01")), ("holdout", sp.index >= pd.Timestamp("2024-01-01"))):
        rows.append({"signal": name, "period": per, "top20%_ex13%": top[sl].mean() * 100,
                     "bottom20%_ex13%": bot[sl].mean() * 100, "spread%": sp[sl].mean() * 100, "t": nw_t(sp[sl])})
# combo: strong stock in strong industry
b = a.dropna(subset=["m12_1"]).copy()
b["ind_q"] = b.groupby("week").ind_r13.rank(pct=True); b["own_q"] = b.groupby("week").m12_1.rank(pct=True)
both = b[(b.ind_q > 0.8) & (b.own_q > 0.8)].groupby("week").ex13.mean()
for per, sl in (("dev", both.index < pd.Timestamp("2024-01-01")), ("holdout", both.index >= pd.Timestamp("2024-01-01"))):
    rows.append({"signal": "strong stock (12-1 top 20%) in strong industry (top 20%)", "period": per,
                 "top20%_ex13%": both[sl].mean() * 100, "t": nw_t(both[sl])})
print("\nC. does industry strength carry forward? next-13-week excess return (%)")
print(pd.DataFrame(rows).round(2).to_string(index=False))
