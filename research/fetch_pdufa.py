"""
research/fetch_pdufa.py -- build a historical FDA-decision (PDUFA) calendar from SEC filings.

EDGAR full-text search for "PDUFA" in 8-K/10-Q/10-K/S-1 documents (2016+), then read each
hit and pull every "PDUFA ... <date>" mention with the filing date it was first public.
Output: research_data/biotech/pdufa_mentions.parquet
  cik, company, form, filed, pdufa_date, snippet   (one row per distinct mention)
"""
import json, os, re, threading, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

OUT = Path(os.environ.get("RESEARCH_DATA", "research_data")) / "biotech"
UA = os.environ.get("SEC_USER_AGENT") or "MomentumAlpha research momentum-bot@users.noreply.github.com"
LOCK, LAST = threading.Lock(), [0.0]
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE = re.compile(rf"((?:{MONTHS})\s+\d{{1,2}},\s+20\d\d)")
PDUFA = re.compile(r"PDUFA", re.I)


def get(url, tries=4):
    for i in range(tries):
        with LOCK:
            w = LAST[0] + 0.13 - time.monotonic()
            if w > 0:
                time.sleep(w)
            LAST[0] = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            time.sleep(3 * (i + 1))
    return None


def search(start, end):
    hits, frm = [], 0
    while True:
        q = urllib.parse.urlencode({"q": '"PDUFA"', "dateRange": "custom", "startdt": start, "enddt": end,
                                    "forms": "8-K,10-Q,10-K,S-1,424B4,424B5", "from": frm})
        raw = get(f"https://efts.sec.gov/LATEST/search-index?{q}")
        if not raw:
            break
        j = json.loads(raw)
        hh = j.get("hits", {}).get("hits", [])
        hits += hh
        frm += len(hh)
        if not hh or frm >= min(j["hits"]["total"]["value"], 9900):
            break
    return hits


def main():
    import concurrent.futures as cf
    OUT.mkdir(parents=True, exist_ok=True)
    months = pd.period_range("2016-01", pd.Timestamp.today().strftime("%Y-%m"), freq="M")
    hits = []
    for m in months:
        h = search(m.start_time.strftime("%Y-%m-%d"), m.end_time.strftime("%Y-%m-%d"))
        hits += h
    docs = []
    for h in hits:
        s = h["_source"]; adsh = h["_id"].split(":")[0]; fn = h["_id"].split(":")[1]
        cik = int(s["ciks"][0]) if s.get("ciks") else None
        if cik:
            docs.append({"cik": cik, "company": (s.get("display_names") or [""])[0], "form": s.get("form"),
                         "filed": s.get("file_date"), "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{fn}"})
    docs = pd.DataFrame(docs).drop_duplicates("url")
    print(f"search hits: {len(hits):,}; documents {len(docs):,}", flush=True)

    def scan(d):
        h = get(d["url"])
        if not h:
            return []
        t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h))
        out = []
        for m in PDUFA.finditer(t):
            win = t[m.start(): m.start() + 260]
            dm = DATE.search(win)
            if dm:
                out.append({**{k: d[k] for k in ("cik", "company", "form", "filed")},
                            "pdufa_date": dm.group(1), "snippet": t[max(0, m.start() - 120): m.start() + 260]})
        return out
    rows = []
    with cf.ThreadPoolExecutor(6) as ex:
        for r in ex.map(scan, docs.to_dict("records")):
            rows += r
    df = pd.DataFrame(rows)
    df["pdufa_date"] = pd.to_datetime(df.pdufa_date, errors="coerce")
    df["filed"] = pd.to_datetime(df.filed, errors="coerce")
    df = df.dropna(subset=["pdufa_date"]).drop_duplicates(["cik", "filed", "pdufa_date"])
    df = df[(df.pdufa_date >= df.filed) & (df.pdufa_date <= df.filed + pd.Timedelta(days=500))]
    df.to_parquet(OUT / "pdufa_mentions.parquet", index=False)
    print(f"::notice::PDUFA mentions: {len(df):,} from {df.cik.nunique():,} companies; "
          f"distinct (cik, PDUFA date) {df[['cik', 'pdufa_date']].drop_duplicates().shape[0]:,}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-10:]:
            print(f"::error::{line}")
        raise
