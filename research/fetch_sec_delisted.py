"""
research/fetch_sec_delisted.py -- SEC filings for companies that have since DELISTED (Oct 17, 2026).

Why: sec/filings_*.parquet came from the SEC's current company_tickers list, so every company
that later delisted is missing (about 1,500 of the ~6,600 stocks trading in 2018). That flatters
every 2017-2023 filing-based test (contract wins, 8-K items, LLM reader, precursors).

How:
  1. inactive Alpaca symbols with price history (assets.parquet: status inactive, has_data)
  2. EDGAR quarterly form.idx 2016Q1..2024Q4 -> every filer's name and CIK
  3. match by normalised company name (share-class words and corporate suffixes removed);
     only unique matches, and only CIKs NOT already in the current-ticker set
  4. that CIK's submissions JSON (fetch_sec._issuer) -> same schema as filings_<yr>.parquet
Output: research_data/sec/filings_delisted.parquet (+ sec/delisted_map.csv for audit)
"""
import concurrent.futures as cf
import io, json, os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import fetch_sec as F

RD = os.environ.get("RESEARCH_DATA", "research_data")
CLASS = re.compile(r"\b(CLASS [A-Z]|ORDINARY SHARES?|COMMON STOCK|COMMON SHARES?|AMERICAN DEPOSITARY SHARES?|"
                   r"ADS|ADR|SHARES? OF BENEFICIAL INTEREST|NEW|UNITS?|WARRANTS?|RIGHTS?|PAR VALUE.*|"
                   r"\$?\d+(\.\d+)? PAR.*|SERIES [A-Z])\b")
SUFFIX = re.compile(r"\b(THE|INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LLC|L L C|LTD|LIMITED|PLC|LP|L P|"
                    r"HOLDINGS?|GROUP|N V|NV|SA|S A|AG|SE|BANCORP|/[A-Z]{2}/?)\b")


def norm(s):
    s = str(s).upper().replace("&", " AND ")
    s = re.sub(r"/[A-Z]{2}/?$", " ", s)                      # EDGAR state tags: "ACME CORP /DE/"
    s = CLASS.sub(" ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def edgar_names():
    rows = []
    for y in range(2016, 2025):
        for q in (1, 2, 3, 4):
            raw = F.get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/company.idx", binary=True)
            if not raw:
                print(f"  index {y}Q{q} missing"); continue
            txt = raw.decode("latin-1").splitlines()
            start = next(i for i, l in enumerate(txt) if l.startswith("-----")) + 1
            for l in txt[start:]:
                # fixed-width: Company Name | Form Type | CIK | Date Filed | File Name
                m = re.match(r"^(.*?)\s{2,}(\S.*?)\s{2,}(\d+)\s+(\d{4}-\d{2}-\d{2})\s+(\S+)$", l)
                if m and m.group(2) in ("8-K", "10-K", "10-Q", "20-F", "6-K", "S-1", "424B4"):
                    rows.append((m.group(1).strip(), int(m.group(3))))
        print(f"  index {y}: {len(rows):,} rows", flush=True)
    d = pd.DataFrame(rows, columns=["name", "cik"]).drop_duplicates()
    d["key"] = d.name.map(norm)
    return d


def main():
    a = pd.read_parquet(f"{RD}/assets.parquet")
    a = a[(a.status == "inactive") & a.has_data.astype(bool)].copy()
    a["key"] = a.name.map(norm)
    cur = json.loads(F.get("https://www.sec.gov/files/company_tickers.json"))
    cur_cik = {int(v["cik_str"]) for v in cur.values()}
    e = edgar_names()
    e = e[~e.cik.isin(cur_cik) & (e.key.str.len() >= 4)]
    ciks_per_key = e.groupby("key").cik.nunique()
    uniq = e[e.key.isin(ciks_per_key[ciks_per_key == 1].index)].drop_duplicates("key")
    m = a.merge(uniq[["key", "cik", "name"]].rename(columns={"name": "edgar_name"}), on="key", how="left")
    m.to_csv(f"{RD}/sec/delisted_map.csv", index=False)
    hit = m.dropna(subset=["cik"])
    hit = hit.drop_duplicates("cik")
    print(f"::notice::delisted map: {len(a):,} inactive symbols with data, {len(hit):,} matched to a unique "
          f"non-current CIK ({len(hit) / len(a) * 100:.0f}%)")
    rows, done = [], 0
    with cf.ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(F._issuer, int(c), s): s for c, s in zip(hit.cik, hit.symbol)}
        for fu in cf.as_completed(futs):
            try:
                rows.extend(fu.result())
            except Exception as ex_:
                print(f"  {futs[fu]}: {ex_}")
            done += 1
            if done % 250 == 0:
                print(f"  {done:,}/{len(futs):,} issuers, {len(rows):,} filings", flush=True)
    df = pd.DataFrame(rows)
    df["accepted_et"] = (pd.to_datetime(df.accepted, utc=True, errors="coerce")
                         .dt.tz_convert("America/New_York").dt.tz_localize(None))
    df = df.drop(columns="accepted")
    df["filing_date"] = pd.to_datetime(df.filing_date)
    df.to_parquet(f"{RD}/sec/filings_delisted.parquet", index=False, compression="zstd")
    k8 = df[df.form == "8-K"]
    print(f"::notice::delisted filings: {len(df):,} filings from {df.cik.nunique():,} issuers; 8-Ks {len(k8):,} "
          f"({k8.filing_date.dt.year.value_counts().sort_index().to_dict()})")


if __name__ == "__main__":
    main()
