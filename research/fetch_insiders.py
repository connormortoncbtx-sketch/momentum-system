"""
research/fetch_insiders.py
==========================
Open-market insider PURCHASES (Form 4, transaction code 'P') from SEC's
quarterly Insider Transactions Data Sets. Only purchases: sales are mostly
diversification/taxes and carry little signal; grants/exercises carry none.

Output research_data/insider_buys.parquet, one row per purchase:
    symbol, filing_date, trans_date, owner_cik, is_officer, is_director,
    is_tenpct, title, shares, price, value_usd
Usable for trading from filing_date (when the market learns of it).

Runs in GitHub Actions (research_insiders.yml).
"""
import io
import logging
import os
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

log = logging.getLogger("fetch_insiders")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
UA = os.environ.get("SEC_USER_AGENT") or "MomentumAlpha research momentum-bot@users.noreply.github.com"
INDEX = "https://www.sec.gov/dera/data/form-345"


def get(url, binary=False, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                time.sleep(0.2)
                data = r.read()
                return data if binary else data.decode("utf-8", "replace")
        except Exception as e:
            log.warning(f"  {url}: {e}")
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"failed: {url}")


def quarter_links(start_year=2016):
    html = get(INDEX)
    links = sorted(set(re.findall(r'href="([^"]*?(\d{4})q(\d)_form345\.zip)"', html)))
    out = []
    for href, yr, q in links:
        if int(yr) >= start_year:
            url = href if href.startswith("http") else "https://www.sec.gov" + href
            out.append((int(yr), int(q), url))
    log.info(f"{len(out)} quarterly files since {start_year}")
    return out


def parse_quarter(blob: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(blob))
    names = {n.split("/")[-1].upper(): n for n in z.namelist()}
    rd = lambda n, cols: pd.read_csv(z.open(names[n]), sep="\t", dtype=str, usecols=lambda c: c in cols,
                                     on_bad_lines="skip", low_memory=False)
    sub = rd("SUBMISSION.TSV", {"ACCESSION_NUMBER", "FILING_DATE", "ISSUERTRADINGSYMBOL", "DOCUMENT_TYPE"})
    own = rd("REPORTINGOWNER.TSV", {"ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"})
    tr = rd("NONDERIV_TRANS.TSV", {"ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                    "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"})
    tr = tr[(tr.TRANS_CODE == "P") & (tr.TRANS_ACQUIRED_DISP_CD == "A")]
    sub = sub[sub.DOCUMENT_TYPE.isin(["4", "4/A"])]
    own = own.drop_duplicates("ACCESSION_NUMBER")      # first reporting owner per filing
    df = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
    rel = df.RPTOWNER_RELATIONSHIP.fillna("").str.upper()
    out = pd.DataFrame({
        "symbol": df.ISSUERTRADINGSYMBOL.str.upper().str.strip(),
        "filing_date": pd.to_datetime(df.FILING_DATE, format="%d-%b-%Y", errors="coerce"),
        "trans_date": pd.to_datetime(df.TRANS_DATE, format="%d-%b-%Y", errors="coerce"),
        "owner_cik": df.RPTOWNERCIK,
        "is_officer": rel.str.contains("OFFICER"),
        "is_director": rel.str.contains("DIRECTOR"),
        "is_tenpct": rel.str.contains("TENPERCENT|10%", regex=True),
        "title": df.RPTOWNER_TITLE.fillna(""),
        "shares": pd.to_numeric(df.TRANS_SHARES, errors="coerce"),
        "price": pd.to_numeric(df.TRANS_PRICEPERSHARE, errors="coerce"),
    })
    out["value_usd"] = out.shares * out.price
    return out.dropna(subset=["symbol", "filing_date", "value_usd"])


def main():
    data = Path(os.environ.get("RESEARCH_DATA", "research_data"))
    frames = []
    for yr, q, url in quarter_links():
        try:
            frames.append(parse_quarter(get(url, binary=True)))
            log.info(f"  {yr}Q{q}: {len(frames[-1]):,} purchases")
        except Exception as e:
            log.warning(f"  {yr}Q{q} failed: {e}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    df.to_parquet(data / "insider_buys.parquet", index=False)
    log.info(f"Wrote {len(df):,} insider purchases, {df.symbol.nunique():,} symbols, "
             f"{df.filing_date.min():%Y-%m} -> {df.filing_date.max():%Y-%m}")
    (data / "README_insiders.md").write_text(f"insider buys: {len(df)} rows\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
