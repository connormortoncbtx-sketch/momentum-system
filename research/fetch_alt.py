"""
research/fetch_alt.py
=====================
Broad-coverage alternative data for the multi-source signal study (Oct 2026).
Every row carries the date it became public, so tests can avoid look-ahead.

  shortvol  FINRA Reg SHO daily short-sale volume PER SYMBOL (all NMS stocks),
            2016+, aggregated to weeks (Mon-Fri) -> alt/shortvol_<yr>.parquet
            columns: week (Friday), symbol, short_volume, total_volume, days
            public: next morning (FINRA posts each day's file that evening)
  ftd       SEC fails-to-deliver, per symbol, twice-monthly files 2016+
            -> alt/ftd.parquet: settle_date, symbol, fails, price, avail_date
            avail_date = end of the half-month + 15 days (SEC posts with a lag;
            conservative so tests never use a file before it existed)
  wiki      English-Wikipedia daily page views per company article (2015-07+).
            Ticker -> article from Wikidata (NYSE/Nasdaq listings with ticker
            qualifier). -> alt/wiki_map.parquet, alt/wiki_<yr>.parquet
            (date, symbol, views). public: next day

Env: ONLY=shortvol,ftd,wiki  START=2016-01-01  SEC_USER_AGENT
"""
import concurrent.futures as cf
import datetime as dt
import io
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

log = logging.getLogger("fetch_alt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
UA = os.environ.get("SEC_USER_AGENT") or "MomentumAlpha research momentum-bot@users.noreply.github.com"
START = os.environ.get("START") or "2016-01-01"
OUT = Path(os.environ.get("RESEARCH_DATA", "research_data")) / "alt"


class Limiter:
    def __init__(self, per_sec):
        self.gap, self.lock, self.t = 1.0 / per_sec, threading.Lock(), 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.t:
                time.sleep(self.t - now)
            self.t = max(now, self.t) + self.gap


LIM = {"finra": Limiter(8), "sec": Limiter(8), "wiki": Limiter(15)}


def get(url, kind, binary=False, tries=4, timeout=60, headers=None):
    for i in range(tries):
        LIM[kind].wait()
        try:
            h = {"User-Agent": UA, "Accept-Encoding": "identity"}
            h.update(headers or {})
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                data = r.read()
                return data if binary else data.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(4 * (i + 1))
        except Exception:
            time.sleep(4 * (i + 1))
    log.warning(f"  gave up: {url[:120]}")
    return None


def save_by_year(df, name, date_col):
    for yr, part in df.groupby(df[date_col].dt.year):
        f = OUT / f"{name}_{int(yr)}.parquet"
        part.to_parquet(f, index=False, compression="zstd")
        log.info(f"  wrote {f.name}: {len(part):,} rows, {f.stat().st_size / 1e6:.1f} MB")


# ── FINRA daily short volume, per symbol ─────────────────────────────────────
def _finra_day(d):
    txt = get(f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d:%Y%m%d}.txt", "finra", timeout=30)
    if not txt:
        return None
    df = pd.read_csv(io.StringIO(txt), sep="|", dtype={"Symbol": str}, usecols=lambda c: c in
                     ("Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume"))
    df = df[pd.to_numeric(df.get("TotalVolume"), errors="coerce").notna()]
    df["date"] = pd.Timestamp(d)
    return df.rename(columns={"Symbol": "symbol", "ShortVolume": "short_volume", "TotalVolume": "total_volume"})[
        ["date", "symbol", "short_volume", "total_volume"]]


def fetch_shortvol():
    """Fetch day files year by year and aggregate to weeks as we go (memory-safe)."""
    total, syms = 0, set()
    for yr in range(int(START[:4]), dt.date.today().year + 1):
        days = pd.bdate_range(max(pd.Timestamp(START), pd.Timestamp(yr, 1, 1)),
                              min(pd.Timestamp(dt.date.today() - dt.timedelta(days=1)), pd.Timestamp(yr, 12, 31)))
        frames = []
        with cf.ThreadPoolExecutor(8) as ex:
            for r in ex.map(_finra_day, days):
                if r is not None:
                    frames.append(r)
        if not frames:
            continue
        d = pd.concat(frames, ignore_index=True)
        d["short_volume"] = pd.to_numeric(d.short_volume, errors="coerce")
        d["total_volume"] = pd.to_numeric(d.total_volume, errors="coerce")
        d["week"] = d.date + pd.to_timedelta(4 - d.date.dt.weekday, unit="D")
        w = d.groupby(["week", "symbol"]).agg(short_volume=("short_volume", "sum"),
                                              total_volume=("total_volume", "sum"),
                                              days=("date", "nunique")).reset_index()
        for c in ("short_volume", "total_volume"):
            w[c] = w[c].astype("float32")
        f = OUT / f"shortvol_{yr}.parquet"
        w.to_parquet(f, index=False, compression="zstd")
        log.info(f"  FINRA {yr}: {len(frames)} days, {len(w):,} symbol-weeks, {f.stat().st_size / 1e6:.1f} MB")
        total += len(w); syms |= set(w.symbol)
    return total, len(syms)


# ── SEC fails-to-deliver ─────────────────────────────────────────────────────
def ftd_links():
    html = ""
    for page in ("https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data",
                 "https://www.sec.gov/data/foiadocsfailsdatahtm"):
        html = get(page, "sec") or ""
        if "cnsfails" in html:
            break
    links = sorted(set(re.findall(r'href="([^"]*cnsfails(\d{6})([ab])\.zip)"', html)))
    out = [(("https://www.sec.gov" + h) if h.startswith("/") else h, ym, half) for h, ym, half in links
           if ym >= START[:4] + START[5:7]]
    if not out:      # documented pattern fallback
        for p in pd.period_range(START, dt.date.today(), freq="M"):
            for half in "ab":
                out.append((f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{p.strftime("%Y%m")}{half}.zip",
                            p.strftime("%Y%m"), half))
    return out


def fetch_ftd():
    frames = []
    for url, ym, half in ftd_links():
        blob = get(url, "sec", binary=True, timeout=120)
        if not blob:
            continue
        try:
            z = zipfile.ZipFile(io.BytesIO(blob))
            raw = z.read(z.namelist()[0]).decode("latin-1")
            df = pd.read_csv(io.StringIO(raw), sep="|", dtype=str, on_bad_lines="skip")
        except Exception as e:
            log.warning(f"  {ym}{half}: {e}")
            continue
        df.columns = [c.strip().upper() for c in df.columns]
        df = df.rename(columns={"SETTLEMENT DATE": "settle_date", "SYMBOL": "symbol",
                                "QUANTITY (FAILS)": "fails", "PRICE": "price"})
        df = df[["settle_date", "symbol", "fails", "price"]].dropna(subset=["symbol"])
        df["settle_date"] = pd.to_datetime(df.settle_date, format="%Y%m%d", errors="coerce")
        df["fails"] = pd.to_numeric(df.fails, errors="coerce").astype("float32")
        df["price"] = pd.to_numeric(df.price, errors="coerce").astype("float32")
        y, m = int(ym[:4]), int(ym[4:])
        half_end = pd.Timestamp(y, m, 15) if half == "a" else pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
        df["avail_date"] = half_end + pd.Timedelta(days=15)
        frames.append(df.dropna(subset=["settle_date"]))
        log.info(f"  FTD {ym}{half}: {len(df):,} rows")
    d = pd.concat(frames, ignore_index=True)
    d.to_parquet(OUT / "ftd.parquet", index=False, compression="zstd")
    return len(d), d.symbol.nunique()


# ── Wikipedia page views ─────────────────────────────────────────────────────
SPARQL = """
SELECT ?item ?ticker ?exch ?article WHERE {
  VALUES ?exch { wd:Q13677 wd:Q82059 }
  ?item p:P414 ?st . ?st ps:P414 ?exch . ?st pq:P249 ?ticker .
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
}"""


def wiki_map():
    url = "https://query.wikidata.org/sparql?format=json&query=" + urllib.parse.quote(SPARQL)
    raw = get(url, "wiki", timeout=120, headers={"Accept": "application/sparql-results+json"})
    rows = []
    for b in json.loads(raw)["results"]["bindings"]:
        title = urllib.parse.unquote(b["article"]["value"].rsplit("/", 1)[-1])
        rows.append({"symbol": b["ticker"]["value"].upper().replace("-", "."), "title": title,
                     "wikidata": b["item"]["value"].rsplit("/", 1)[-1],
                     "exchange": "NYSE" if b["exch"]["value"].endswith("Q13677") else "NASDAQ"})
    m = pd.DataFrame(rows).drop_duplicates(["symbol", "title"])
    m.to_parquet(OUT / "wiki_map.parquet", index=False)
    log.info(f"  Wikidata: {len(m):,} ticker-article pairs, {m.symbol.nunique():,} tickers")
    return m


def _views(title):
    t = urllib.parse.quote(title.replace(" ", "_"), safe="")
    s = START.replace("-", "") if START >= "2015-07-01" else "20150701"
    e = (dt.date.today() - dt.timedelta(days=1)).strftime("%Y%m%d")
    raw = get(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
              f"all-access/user/{t}/daily/{s}/{e}", "wiki", timeout=60)
    if not raw:
        return title, []
    return title, [(i["timestamp"][:8], i["views"]) for i in json.loads(raw).get("items", [])]


def fetch_wiki():
    m = wiki_map()
    titles = sorted(m.title.unique())
    rows, done = [], 0
    with cf.ThreadPoolExecutor(8) as ex:
        for title, pts in ex.map(_views, titles):
            rows.extend((title, d, v) for d, v in pts)
            done += 1
            if done % 500 == 0:
                log.info(f"  wiki {done:,}/{len(titles):,} articles, {len(rows):,} points")
    v = pd.DataFrame(rows, columns=["title", "date", "views"])
    v["date"] = pd.to_datetime(v.date, format="%Y%m%d")
    v["views"] = v.views.astype("int32")
    v = v.merge(m[["title", "symbol"]], on="title")[["date", "symbol", "views"]]
    v = v.groupby(["date", "symbol"], as_index=False).views.sum()     # several articles per ticker -> sum
    save_by_year(v, "wiki", "date")
    return len(v), v.symbol.nunique()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = (os.environ.get("ONLY") or "shortvol,ftd,wiki").split(",")
    notes = []
    for name, fn in (("wiki", fetch_wiki), ("ftd", fetch_ftd), ("shortvol", fetch_shortvol)):
        if name in only:
            try:
                n, k = fn()
                notes.append(f"{name}: {n:,} rows / {k:,} symbols")
            except Exception as e:
                import traceback
                for line in traceback.format_exc().strip().splitlines()[-6:]:
                    print(f"::error::{name}: {line}")
                notes.append(f"{name}: FAILED ({type(e).__name__})")
    print("::notice::alt data: " + "; ".join(notes))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
