"""
research/fetch_10k_sim.py -- "Lazy Prices" input (Cohen, Malloy & Nguyen 2020): how much each 10-K changed vs the same
company's previous 10-K. Nov 15, 2026 (Connor: "test everything").

For every 10-K in research_data/sec/filings_*.parquet + filings_delisted.parquet (2016+), shard SEED of 4 by CIK:
download the primary document from EDGAR (public, SEC_USER_AGENT, <= 8 requests/s per runner), strip HTML, and compare
with the company's previous 10-K:
  sim_cos   cosine similarity of word-count vectors (whole document)
  sim_jac   Jaccard similarity of the word sets
  risk_cos  cosine similarity of Item 1A (risk factors) when both documents have one
Only these numbers are kept (no text is stored). Output: research_data/sec/tenk_sim_<SEED>.parquet.
"""
import collections
import concurrent.futures as cf
import glob
import html
import math
import os
import re
import threading
import time
import urllib.request

import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
UA = os.environ.get("SEC_USER_AGENT") or "MomentumAlpha research momentum-bot@users.noreply.github.com"
SHARD, NSHARD = int(os.environ.get("SEED") or 0), 4
LIMIT = int(os.environ.get("LIMIT") or 0)
TAG = re.compile(r"<[^>]+>")
WORD = re.compile(r"[a-z]{3,}")


class Rate:
    def __init__(self, per_sec):
        self.gap, self.t, self.lock = 1.0 / per_sec, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            self.t = max(self.t + self.gap, now)
            delay = self.t - now
        if delay > 0:
            time.sleep(delay)


RATE = Rate(8)


def get(url, tries=4):
    for k in range(tries):
        RATE.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read(40_000_000).decode("latin-1", errors="ignore")
        except Exception as e:
            if "404" in str(e):
                return None
            time.sleep(2 * (k + 1))
    return None


def text_of(raw):
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    t = html.unescape(TAG.sub(" ", raw)).lower()
    return re.sub(r"\s+", " ", t)


def risk_section(t):
    """Longest span from 'item 1a' to the next 'item 1b' / 'item 2' (skips the table of contents)."""
    best = ""
    for m in re.finditer(r"item\s*1a\.?\s*[-:]?\s*risk\s+factors", t):
        end = re.search(r"item\s*1b\.?|item\s*2\.?\s*[-:]?\s*properties", t[m.end():])
        span = t[m.end(): m.end() + end.start()] if end else ""
        if len(span) > len(best):
            best = span
    return best if len(best) > 2000 else ""


def vec(t):
    return collections.Counter(WORD.findall(t))


def cos(a, b):
    if not a or not b:
        return None
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def jac(a, b):
    if not a or not b:
        return None
    A, B = set(a), set(b)
    return len(A & B) / len(A | B)


def do_cik(cik, rows):
    out, prev, prev_r = [], None, None
    for r in rows.itertuples():
        acc = r.accession.replace("-", "")
        raw = get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{r.primary_doc}")
        if raw is None:
            continue
        t = text_of(raw)
        v, rv = vec(t), vec(risk_section(t))
        out.append(dict(cik=int(cik), accession=r.accession, tickers=r.tickers, filing_date=r.filing_date,
                        accepted_et=r.accepted_et, words=sum(v.values()), risk_words=sum(rv.values()),
                        sim_cos=cos(v, prev) if prev else None, sim_jac=jac(v, prev) if prev else None,
                        risk_cos=cos(rv, prev_r) if (prev_r and rv) else None))
        prev, prev_r = v, (rv or prev_r)
    return out


def main():
    files = sorted(glob.glob(f"{RD}/sec/filings_20*.parquet")) + [f"{RD}/sec/filings_delisted.parquet"]
    F = pd.concat((pd.read_parquet(f, columns=["cik", "tickers", "form", "filing_date", "primary_doc", "accession", "accepted_et"])
                   for f in files if os.path.exists(f)), ignore_index=True)
    F = F[(F.form == "10-K") & F.primary_doc.fillna("").str.lower().str.contains(r"\.htm|\.txt")]
    F = F.drop_duplicates("accession").sort_values(["cik", "filing_date"])
    F = F[F.cik.astype(int) % NSHARD == SHARD]
    ciks = list(F.groupby("cik"))
    if LIMIT:
        ciks = ciks[:LIMIT]
    print(f"shard {SHARD}/{NSHARD}: {sum(len(g) for _, g in ciks):,} 10-Ks from {len(ciks):,} companies", flush=True)
    rows, done, t0 = [], 0, time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        for res in ex.map(lambda x: do_cik(*x), ciks):
            rows += res
            done += 1
            if done % 200 == 0:
                print(f"  {done:,}/{len(ciks):,} companies, {len(rows):,} docs, {time.time() - t0:.0f}s", flush=True)
    D = pd.DataFrame(rows)
    os.makedirs(f"{RD}/sec", exist_ok=True)
    D.to_parquet(f"{RD}/sec/tenk_sim_{SHARD}.parquet", index=False)
    print(f"::notice::10-K similarity shard {SHARD}: {len(D):,} docs, {D.sim_cos.notna().sum():,} with a prior 10-K, "
          f"{D.risk_cos.notna().sum():,} with risk sections")


if __name__ == "__main__":
    main()
