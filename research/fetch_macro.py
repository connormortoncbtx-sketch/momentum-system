"""
research/fetch_macro.py
=======================
Free regime data, saved to research_data/macro/ (runs in GitHub Actions):

  fred_<ID>.parquet     FRED series via fredgraph.csv (no key needed)
  french_*.parquet      Ken French daily factors incl. momentum (UMD), 1926+
  cboe_<NAME>.parquet   CBOE index history (VIX, VIX3M, VIX9D, SKEW)
  cboe_pc_*.parquet     CBOE put/call ratio archive (2006-2019)
  finra_short.parquet   FINRA daily short-sale volume, market-wide (2009+)

Every file keeps the observation date; publication lags are applied in the
analysis scripts, not here.
"""
import concurrent.futures as cf
import datetime as dt
import io
import logging
import os
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

log = logging.getLogger("fetch_macro")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
UA = os.environ.get("SEC_USER_AGENT") or "MomentumAlpha research momentum-bot@users.noreply.github.com"

FRED = {
    # rates / curve
    "DGS10": "10y Treasury (daily)", "DGS2": "2y Treasury (daily)", "DTB3": "3m T-bill (daily)",
    "GS10": "10y Treasury (monthly, 1953+)", "TB3MS": "3m T-bill (monthly, 1934+)",
    "FEDFUNDS": "Fed funds (monthly, 1954+)", "DFII10": "10y real yield (daily, 2003+)",
    # credit
    "BAA": "Moody's Baa yield (monthly, 1919+)", "AAA": "Moody's Aaa yield (monthly, 1919+)",
    "BAA10Y": "Baa minus 10y (daily, 1986+)", "BAMLH0A0HYM2": "HY OAS (daily, 1996+)",
    # conditions / activity / inflation
    "NFCI": "Chicago Fed financial conditions (weekly, 1971+)", "ICSA": "Initial claims (weekly, 1967+)",
    "CPIAUCSL": "CPI (monthly, 1947+)", "UNRATE": "Unemployment (monthly, 1948+)",
    "INDPRO": "Industrial production (monthly, 1919+)", "USREC": "NBER recession flag (monthly)",
    "VIXCLS": "VIX close (daily, 1990+)", "T10Y2Y": "10y-2y (daily, 1976+)", "T10Y3M": "10y-3m (daily, 1982+)",
}
CBOE = {
    "VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "VIX3M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
    "VIX9D": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
    "SKEW": "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv",
}
CBOE_PC = {
    "equity": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv",
    "total": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv",
    "index": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpc.csv",
}
FRENCH = {
    "factors": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
    "momentum": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
}


def get(url, tries=4, binary=False, timeout=60):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                b = r.read()
                return b if binary else b.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (i + 1))
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def fred(out):
    for sid, desc in FRED.items():
        txt = get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
        if not txt:
            log.warning(f"  FRED {sid}: failed"); continue
        df = pd.read_csv(io.StringIO(txt))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        df.to_parquet(out / f"fred_{sid}.parquet", index=False)
        log.info(f"  FRED {sid:<14} {len(df):>6} rows {df.date.min():%Y-%m}..{df.date.max():%Y-%m}  {desc}")


def french(out):
    for name, url in FRENCH.items():
        b = get(url, binary=True)
        if not b:
            log.warning(f"  French {name}: failed"); continue
        z = zipfile.ZipFile(io.BytesIO(b))
        raw = z.read(z.namelist()[0]).decode("latin-1").splitlines()
        rows = [l.split(",") for l in raw if l[:8].strip().isdigit() and len(l.strip()) > 8]
        hdr_idx = next(i for i, l in enumerate(raw) if l.strip() and not l[:1].isdigit() and "," in l
                       and (i + 1 < len(raw) and raw[i + 1][:8].strip().isdigit()))
        cols = ["date"] + [c.strip() for c in raw[hdr_idx].split(",")[1:]]
        df = pd.DataFrame([r[:len(cols)] for r in rows], columns=cols)
        df["date"] = pd.to_datetime(df["date"].str.strip(), format="%Y%m%d")
        for c in cols[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
        df.to_parquet(out / f"french_{name}.parquet", index=False)
        log.info(f"  French {name:<10} {len(df):>6} rows {df.date.min():%Y-%m}..{df.date.max():%Y-%m} cols {cols[1:]}")


def cboe(out):
    for name, url in CBOE.items():
        txt = get(url)
        if not txt:
            log.warning(f"  CBOE {name}: failed"); continue
        df = pd.read_csv(io.StringIO(txt))
        df.columns = [c.strip().lower() for c in df.columns]
        dcol = df.columns[0]
        vcol = "close" if "close" in df.columns else df.columns[-1]
        df = pd.DataFrame({"date": pd.to_datetime(df[dcol], errors="coerce"),
                           "value": pd.to_numeric(df[vcol], errors="coerce")}).dropna()
        df.to_parquet(out / f"cboe_{name}.parquet", index=False)
        log.info(f"  CBOE {name:<6} {len(df):>6} rows {df.date.min():%Y-%m}..{df.date.max():%Y-%m}")
    for name, url in CBOE_PC.items():
        txt = get(url)
        if not txt:
            log.warning(f"  CBOE put/call {name}: failed"); continue
        lines = txt.splitlines()
        start = next(i for i, l in enumerate(lines) if l.upper().startswith("DATE"))
        df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
        df.columns = [c.strip().lower().replace("/", "_").replace(" ", "_") for c in df.columns]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        pc = [c for c in df.columns if "ratio" in c or c in ("p_c_ratio",)]
        df = df.dropna(subset=["date"])
        df.to_parquet(out / f"cboe_pc_{name}.parquet", index=False)
        log.info(f"  CBOE put/call {name:<6} {len(df):>5} rows {df.date.min():%Y-%m}..{df.date.max():%Y-%m} cols {list(df.columns)}")


def finra_day(d):
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d:%Y%m%d}.txt"
    txt = get(url, tries=3, timeout=30)
    if not txt:
        return None
    df = pd.read_csv(io.StringIO(txt), sep="|", dtype={"Symbol": str})
    df = df[pd.to_numeric(df.get("ShortVolume"), errors="coerce").notna()]
    sv, tv = pd.to_numeric(df.ShortVolume), pd.to_numeric(df.TotalVolume)
    return {"date": pd.Timestamp(d), "short_volume": float(sv.sum()), "total_volume": float(tv.sum()),
            "n_symbols": int(len(df))}


def finra(out, start="2009-08-03"):
    days = pd.bdate_range(start, dt.date.today() - dt.timedelta(days=1))
    rows = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(finra_day, days)):
            if r:
                rows.append(r)
            if i % 500 == 0:
                log.info(f"  FINRA {i}/{len(days)} days, {len(rows)} files")
    df = pd.DataFrame(rows).sort_values("date")
    df["short_ratio"] = df.short_volume / df.total_volume
    df.to_parquet(out / "finra_short.parquet", index=False)
    log.info(f"  FINRA short volume: {len(df)} days {df.date.min():%Y-%m}..{df.date.max():%Y-%m}")


def main():
    out = Path(os.environ.get("RESEARCH_DATA", "research_data")) / "macro"
    out.mkdir(parents=True, exist_ok=True)
    only = os.environ.get("ONLY", "fred,french,cboe,finra").split(",")
    for name, fn in (("fred", fred), ("french", french), ("cboe", cboe), ("finra", finra)):
        if name in only:
            log.info(f"== {name}")
            try:
                fn(out)
            except Exception as e:
                log.error(f"{name} failed: {e}")
                print(f"::warning::{name} failed: {e}")
    files = sorted(p.name for p in out.glob("*.parquet"))
    print(f"::notice::macro files: {len(files)} -> {', '.join(files)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
