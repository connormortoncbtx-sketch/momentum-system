"""
research/fetch_merger_index.py -- every merger-target filing 2016-2026 from SEC EDGAR's quarterly full index, mapped
to tickers including delisted ones (Nov 15, 2026). Fixes merger_arb.py's 2024-26 hole: the per-company filing set only
covers companies that still have tickers, so targets acquired since 2023 were missing.

1. EDGAR full-index form.idx for every quarter 2016Q1 -> now: rows with form PREM14A, DEFM14A, SC 14D9, SC TO-T,
   SC TO-C (the index lists a tender-offer filing under the subject company too).
2. Ticker for each CIK, using only what was knowable at the time:
   a. SEC's current company_tickers.json (companies still listed), else
   b. a NAME match against the EODHD delisted list, accepted only if that ticker has Alpaca daily bars in the
      30 trading days before the filing (it was trading then). The match never looks at what happened after the
      filing, so it cannot favour completed deals.
Output: research_data/sec/merger_index_events.csv (cik, name, ticker, form, date, how). Public data only.
"""
import glob
import json
import os
import re
import time
import urllib.request

import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
UA = os.environ.get("SEC_USER_AGENT") or "MomentumAlpha research momentum-bot@users.noreply.github.com"
FORMS = {"PREM14A", "DEFM14A", "SC 14D9", "SC TO-T", "SC TO-C"}
SUFFIX = re.compile(r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|HOLDINGS?|GROUP|LP|L P|LLC|SA|NV|AG|"
                    r"THE|CLASS [A-Z]|COMMON STOCK|ORDINARY SHARES?|NEW|DE|/DE/|/MD/|/NV/)\b")


def get(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if "404" in str(e):
                return None
            time.sleep(3 + 5 * i)
    return None


def norm(s):
    s = re.sub(r"[^A-Z0-9 ]", " ", str(s).upper().replace("&", " AND "))
    s = SUFFIX.sub(" ", s)
    return " ".join(s.split())


def main():
    rows = []
    now = pd.Timestamp.today()
    for y in range(2016, now.year + 1):
        for q in range(1, 5):
            if pd.Timestamp(y, 3 * q - 2, 1) > now:
                break
            raw = get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx")
            time.sleep(0.3)
            if not raw:
                print(f"::warning::no index {y}Q{q}"); continue
            for line in raw.decode("latin-1").splitlines():
                form = line[:12].strip()
                if form not in FORMS:
                    continue
                m = re.match(r"^(.*?)\s+(\d+)\s+(\d{4}-\d{2}-\d{2})\s+(\S+)\s*$", line[12:])
                if not m:
                    continue
                rows.append(dict(form=form, name=m.group(1).strip(), cik=int(m.group(2)), date=pd.Timestamp(m.group(3))))
            print(f"  {y}Q{q}: {len(rows):,} rows so far", flush=True)
    E = pd.DataFrame(rows).drop_duplicates()
    print(f"merger-form rows: {len(E):,}; companies {E.cik.nunique():,}", flush=True)
    tk = json.loads(get("https://www.sec.gov/files/company_tickers.json"))
    cur = {int(v["cik_str"]): v["ticker"].upper() for v in tk.values()}
    L = pd.read_csv(f"{RD}/delisted/eodhd_us_delisted.csv")
    L = L[L.Type == "Common Stock"].copy()
    L["n"] = L.Name.map(norm); L["Code"] = L.Code.astype(str).str.upper()
    byname = L.groupby("n").Code.apply(lambda x: sorted(set(x))).to_dict()
    D = pd.concat([pd.read_parquet(f, columns=["symbol", "date"]) for f in glob.glob(f"{RD}/daily/delisted_ohlcv_*.parquet")])
    D["date"] = pd.to_datetime(D.date)
    spans = D.groupby("symbol").date.agg(["min", "max"])
    out = []
    for r in E.itertuples():
        t, how = cur.get(r.cik), "sec_current"
        if not t:
            how = "name_delisted"
            for c in byname.get(norm(r.name), []):
                if c in spans.index and spans.at[c, "min"] <= r.date - pd.Timedelta(days=5) and spans.at[c, "max"] >= r.date - pd.Timedelta(days=45):
                    t = c; break                      # trading in the weeks before the filing (ex ante)
        if t:
            out.append(dict(cik=r.cik, name=r.name, ticker=t, form=r.form, date=r.date.date(), how=how))
    O = pd.DataFrame(out).sort_values("date")
    O.to_csv(f"{RD}/sec/merger_index_events.csv", index=False)
    by = O.assign(y=pd.to_datetime(O.date).dt.year).groupby(["y", "how"]).cik.nunique().unstack(fill_value=0)
    print(by.to_string())
    print(f"::notice::merger index: {len(E):,} filings ({E.cik.nunique():,} companies); mapped {O.cik.nunique():,} companies "
          f"({(O.how == 'name_delisted').groupby(O.cik).any().sum():,} via delisted names)")


if __name__ == "__main__":
    main()
