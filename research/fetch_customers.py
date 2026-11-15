"""
research/fetch_customers.py -- supplier -> major-customer links from 10-K text (Cohen & Frazzini 2008 "economic links"),
Nov 15, 2026. Companies must disclose customers >= 10% of revenue; many name them ("Walmart accounted for 22% of net
sales"). For every 10-K (shard SEED of 4, as fetch_10k_sim.py): download, strip HTML, split into sentences, keep
sentences that mention a percentage AND revenue/sales/customer, and match them against the names of LARGE listed
companies (median daily $ volume >= $50M in the weekly panel; names cleaned of Inc/Corp/etc., >= 5 letters, generic
words excluded). Output: research_data/sec/customer_links_<SEED>.parquet (supplier cik/tickers, filing_date,
customer symbol, pct if parsed, the matched sentence truncated to 300 chars).
"""
import concurrent.futures as cf
import glob
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import fetch_10k_sim as T  # noqa: E402   (rate-limited EDGAR get + html cleaner)

RD = os.environ.get("RESEARCH_DATA", "research_data")
SHARD, NSHARD = int(os.environ.get("SEED") or 0), 4
LIMIT = int(os.environ.get("LIMIT") or 0)
SUFFIX = re.compile(r"\b(inc|corp|corporation|co|company|ltd|limited|plc|holdings?|group|the|class [a-c]|common stock|"
                    r"ordinary shares|n\.?v|s\.?a|ag|se|llc|lp|trust|international|incorporated|&)\b\.?", re.I)
GENERIC = {"target", "apple", "general", "national", "american", "united", "first", "global", "energy", "health", "capital",
           "financial", "systems", "technologies", "industries", "services", "resources", "partners", "brands", "solutions",
           "international", "pacific", "southern", "western", "eastern", "northern", "republic", "public", "power"}
SENT = re.compile(r"[^.;]{0,400}?\d{1,2}(?:\.\d)?\s?%[^.;]{0,400}")
CTX = re.compile(r"revenue|net sales|sales|customer", re.I)


def big_names():
    W = pd.concat((pd.read_parquet(f, columns=["symbol", "dollar_vol", "n_days"]) for f in glob.glob(f"{RD}/weekly_*.parquet")))
    adv = (W.dollar_vol / W.n_days).groupby(W.symbol).median()
    big = adv[adv >= 5e7].index
    names = pd.read_parquet(f"{RD}/assets.parquet").set_index("symbol")["name"].reindex(big).dropna()
    out = {}
    for s, n in names.items():
        c = re.sub(r"\s+", " ", SUFFIX.sub(" ", n.lower().replace(",", " "))).strip()
        if len(c) >= 5 and c not in GENERIC and not c.split()[0] in {"ishares", "spdr", "invesco", "vanguard"}:
            out.setdefault(c, set()).add(s)
    out = {k: v for k, v in out.items() if len(v) == 1}          # ambiguous names dropped
    return {k: next(iter(v)) for k, v in out.items()}


def main():
    names = big_names()
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\b", re.I)
    files = sorted(glob.glob(f"{RD}/sec/filings_20*.parquet")) + [f"{RD}/sec/filings_delisted.parquet"]
    F = pd.concat((pd.read_parquet(f, columns=["cik", "tickers", "form", "filing_date", "primary_doc", "accession", "accepted_et"])
                   for f in files if os.path.exists(f)), ignore_index=True)
    F = F[(F.form == "10-K") & F.primary_doc.fillna("").str.lower().str.contains(r"\.htm|\.txt")].drop_duplicates("accession")
    F = F[F.cik.astype(int) % NSHARD == SHARD].sort_values("filing_date")
    if LIMIT:
        F = F.head(LIMIT)
    print(f"shard {SHARD}: {len(F):,} 10-Ks; {len(names):,} large-company names to match", flush=True)

    def one(r):
        raw = T.get(f"https://www.sec.gov/Archives/edgar/data/{int(r.cik)}/{r.accession.replace('-', '')}/{r.primary_doc}")
        if raw is None:
            return []
        raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
        t = re.sub(r"\s+", " ", T.html.unescape(T.TAG.sub(" ", raw)))          # original case kept
        own = set(str(r.tickers).split(","))
        hits = []
        for m in SENT.finditer(t):
            s = m.group(0)
            if not CTX.search(s):
                continue
            for hit in set(pat.findall(s)):
                nm = hit.lower()
                if " " not in nm and (not hit[0].isupper() or re.search(r"\b" + re.escape(nm) + r"\b", t)):
                    continue      # one-word name: must be capitalised here and never used as an ordinary word
                sym = names[nm]
                if sym in own:
                    continue
                pct = re.search(r"(\d{1,2}(?:\.\d)?)\s?%", s)
                hits.append(dict(cik=int(r.cik), tickers=r.tickers, filing_date=r.filing_date, accepted_et=r.accepted_et,
                                 customer=sym, pct=float(pct.group(1)) if pct else None, sentence=s.strip()[:300]))
        return hits

    rows, t0 = [], time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        for i, h in enumerate(ex.map(one, F.itertuples())):
            rows += h
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1:,}/{len(F):,} docs, {len(rows):,} links, {time.time() - t0:.0f}s", flush=True)
    D = pd.DataFrame(rows).drop_duplicates(["cik", "filing_date", "customer"]) if rows else pd.DataFrame()
    os.makedirs(f"{RD}/sec", exist_ok=True)
    D.to_parquet(f"{RD}/sec/customer_links_{SHARD}.parquet", index=False)
    print(f"::notice::customer links shard {SHARD}: {len(D):,} supplier-customer-year links, "
          f"{D.customer.nunique() if len(D) else 0} customers; top {D.customer.value_counts().head(8).to_dict() if len(D) else {}}")


if __name__ == "__main__":
    main()
