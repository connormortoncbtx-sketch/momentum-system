"""
research/fetch_sec.py
=====================
Point-in-time SEC data for the multi-source signal study (Oct 2026).

  filings      every filing since 2016 by every currently listed issuer
               (SEC submissions API): cik, tickers, form, filing date,
               acceptance time (ET), 8-K item codes, primary document.
               -> research_data/sec/filings_<year>.parquet
  fundamentals SEC Financial Statement Data Sets (quarterly zips, all XBRL
               10-K/10-Q filers, delisted included): selected us-gaap/dei tags,
               current-period values only, with the filing's acceptance time
               so every number is used only after it became public.
               -> research_data/sec/fundamentals_<year>.parquet

Env: ONLY=filings,fundamentals  START=2016  SEC_USER_AGENT (required by SEC)
Known limits: filings covers issuers listed today (delisted names are missing
from that part); fundamentals covers everyone who filed XBRL.
"""
import concurrent.futures as cf
import io
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

log = logging.getLogger("fetch_sec")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
UA = os.environ.get("SEC_USER_AGENT") or "MomentumAlpha research momentum-bot@users.noreply.github.com"
START = int(os.environ.get("START") or 2016)
OUT = Path(os.environ.get("RESEARCH_DATA", "research_data")) / "sec"

TAGS = {
    # income statement (flows)
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
    "GrossProfit", "OperatingIncomeLoss", "NetIncomeLoss", "ResearchAndDevelopmentExpense",
    "EarningsPerShareDiluted", "EarningsPerShareBasic", "InterestExpense", "IncomeTaxExpenseBenefit",
    # cash flow
    "NetCashProvidedByUsedInOperatingActivities", "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsForRepurchaseOfCommonStock", "PaymentsOfDividends", "ProceedsFromIssuanceOfCommonStock",
    "ShareBasedCompensation", "DepreciationDepletionAndAmortization",
    # balance sheet (stocks)
    "Assets", "Liabilities", "StockholdersEquity", "AssetsCurrent", "LiabilitiesCurrent",
    "CashAndCashEquivalentsAtCarryingValue", "LongTermDebtNoncurrent", "LongTermDebt",
    "InventoryNet", "AccountsReceivableNetCurrent", "Goodwill", "RetainedEarningsAccumulatedDeficit",
    "CommonStockSharesOutstanding",
    # cover page
    "EntityCommonStockSharesOutstanding", "EntityPublicFloat",
}


class Limiter:
    def __init__(self, per_sec):
        self.gap, self.lock, self.t = 1.0 / per_sec, threading.Lock(), 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.t:
                time.sleep(self.t - now)
            self.t = max(now, self.t) + self.gap


SEC = Limiter(8)          # SEC fair-access limit is 10 requests/second


def get(url, binary=False, tries=5, timeout=60):
    for i in range(tries):
        SEC.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                return data if binary else data.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            log.warning(f"  HTTP {e.code} {url}")
            time.sleep(5 * (i + 1))
        except Exception as e:
            log.warning(f"  {type(e).__name__} {url}")
            time.sleep(5 * (i + 1))
    log.warning(f"  gave up: {url}")
    return None


# ── filings ──────────────────────────────────────────────────────────────────
def _issuer(cik, tickers):
    import json
    raw = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if not raw:
        return []
    sub = json.loads(raw)
    blocks = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files", []):
        if f.get("filingTo", "9999") >= f"{START}-01-01":
            more = get(f"https://data.sec.gov/submissions/{f['name']}")
            if more:
                blocks.append(json.loads(more))
    out = []
    for b in blocks:
        n = len(b["form"])
        for k in range(n):
            if b["filingDate"][k] < f"{START}-01-01":
                continue
            out.append({"cik": cik, "tickers": tickers, "form": b["form"][k],
                        "filing_date": b["filingDate"][k], "accepted": b["acceptanceDateTime"][k],
                        "items": (b.get("items") or [""] * n)[k] or "",
                        "primary_doc": (b.get("primaryDocument") or [""] * n)[k] or "",
                        "doc_desc": (b.get("primaryDocDescription") or [""] * n)[k] or "",
                        "size": (b.get("size") or [0] * n)[k] or 0,
                        "accession": b["accessionNumber"][k],
                        "sic": sub.get("sic") or "", "exchanges": ",".join(sub.get("exchanges") or [])})
    return out


def fetch_filings():
    import json
    tick = json.loads(get("https://www.sec.gov/files/company_tickers.json"))
    by_cik = {}
    for v in tick.values():
        by_cik.setdefault(int(v["cik_str"]), []).append(v["ticker"].upper().replace("-", "."))
    log.info(f"{len(by_cik):,} issuers ({sum(map(len, by_cik.values())):,} tickers)")
    rows, done = [], 0
    with cf.ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(_issuer, c, ",".join(t)): c for c, t in by_cik.items()}
        for fu in cf.as_completed(futs):
            try:
                rows.extend(fu.result())
            except Exception as e:
                log.warning(f"  issuer {futs[fu]} failed: {e}")
            done += 1
            if done % 1000 == 0:
                log.info(f"  {done:,}/{len(by_cik):,} issuers, {len(rows):,} filings")
    df = pd.DataFrame(rows)
    # acceptanceDateTime is UTC labelled 'Z' (verified in fetch_events.py) -> Eastern
    df["accepted_et"] = (pd.to_datetime(df.accepted, utc=True, errors="coerce")
                         .dt.tz_convert("America/New_York").dt.tz_localize(None))
    df = df.drop(columns="accepted")
    df["filing_date"] = pd.to_datetime(df.filing_date)
    return df


# ── fundamentals (Financial Statement Data Sets) ─────────────────────────────
FSDS = "https://www.sec.gov/dera/data/financial-statement-data-sets"


def fsds_links():
    html = get(FSDS) or ""
    links = sorted(set(re.findall(r'href="([^"]*?(\d{4})q([1-4])\.zip)"', html)))
    out = []
    for href, yr, q in links:
        if int(yr) >= START:
            out.append((int(yr), int(q), href if href.startswith("http") else "https://www.sec.gov" + href))
    if not out:   # fall back to the documented URL pattern
        out = [(y, q, f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{y}q{q}.zip")
               for y in range(START, pd.Timestamp.today().year + 1) for q in range(1, 5)]
    return out


def parse_fsds(blob: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(blob))
    names = {n.split("/")[-1].lower(): n for n in z.namelist()}
    sub = pd.read_csv(z.open(names["sub.txt"]), sep="\t", dtype=str, low_memory=False,
                      usecols=["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed", "accepted"])
    sub = sub[sub.form.isin(["10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F", "10-KT", "10-QT"])]
    keep = []
    for chunk in pd.read_csv(z.open(names["num.txt"]), sep="\t", dtype=str, low_memory=False,
                             usecols=["adsh", "tag", "version", "ddate", "qtrs", "uom", "value", "coreg"],
                             chunksize=1_000_000):
        c = chunk[chunk.tag.isin(TAGS) & chunk.coreg.isna() & chunk.adsh.isin(sub.adsh)]
        keep.append(c.drop(columns="coreg"))
    num = pd.concat(keep, ignore_index=True)
    num = num.merge(sub[["adsh", "period"]], on="adsh")
    # current-period values only (balance sheet at period end; flows ending at period end)
    num = num[num.ddate == num.period].drop(columns="period")
    num["value"] = pd.to_numeric(num.value, errors="coerce")
    num["qtrs"] = pd.to_numeric(num.qtrs, errors="coerce").astype("Int8")
    return num.merge(sub, on="adsh")


def fetch_fundamentals():
    frames = []
    for yr, q, url in fsds_links():
        blob = get(url, binary=True, timeout=300)
        if not blob:
            log.info(f"  {yr}q{q}: not available")
            continue
        try:
            d = parse_fsds(blob)
        except Exception as e:
            log.warning(f"  {yr}q{q}: parse failed {e}")
            continue
        frames.append(d)
        log.info(f"  {yr}q{q}: {len(d):,} values, {d.adsh.nunique():,} filings")
    df = pd.concat(frames, ignore_index=True)
    df["cik"] = df.cik.astype(int)
    df["accepted_et"] = pd.to_datetime(df.accepted, errors="coerce")   # FSDS 'accepted' is already ET
    for c in ("filed", "period"):
        df[c] = pd.to_datetime(df[c], format="%Y%m%d", errors="coerce")
    df = df.drop(columns=["accepted", "ddate", "name", "version", "fy"], errors="ignore")
    for c in ("tag", "uom", "form", "fp", "sic"):
        df[c] = df[c].astype("category")
    return df


def save_by_year(df, name, date_col):
    """One file per year keeps each under GitHub's 100 MB limit."""
    for yr, part in df.groupby(df[date_col].dt.year):
        f = OUT / f"{name}_{int(yr)}.parquet"
        part.to_parquet(f, index=False, compression="zstd")
        log.info(f"  wrote {f.name}: {len(part):,} rows, {f.stat().st_size / 1e6:.1f} MB")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = (os.environ.get("ONLY") or "filings,fundamentals").split(",")
    if "fundamentals" in only:
        fu = fetch_fundamentals()
        save_by_year(fu, "fundamentals", "filed")
        log.info(f"fundamentals: {len(fu):,} values, {fu.cik.nunique():,} companies")
        print(f"::notice::sec fundamentals: {len(fu):,} values, {fu.cik.nunique():,} companies, "
              f"{fu.filed.min():%Y-%m}..{fu.filed.max():%Y-%m}")
    if "filings" in only:
        fi = fetch_filings()
        save_by_year(fi, "filings", "filing_date")
        top = fi.form.value_counts().head(12).to_dict()
        print(f"::notice::sec filings: {len(fi):,} filings, {fi.cik.nunique():,} issuers; top forms {top}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
