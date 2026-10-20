"""Probe: ForecastEx (Interactive Brokers' event exchange) public daily files -- is there a thin, wide-spread,
automatable book like Kalshi weather? Pulls summary / prices / pairs CSVs for the last ~60 days
(https://www.forecastex.com/api/download?type=...&date=YYYYMMDD), saves them raw under research_data/forecastex/,
and prints the schema plus activity by product. Public data only, no account."""
import datetime as dt, io, os, traceback, urllib.request
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = f"{RD}/forecastex"; os.makedirs(OUT, exist_ok=True)
URL = "https://www.forecastex.com/api/download?type={t}&date={d}"
UA = {"User-Agent": "Mozilla/5.0 (research probe)", "Accept": "text/csv,*/*"}


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def main():
    got = {"summary": [], "prices": [], "pairs": []}
    today = dt.date.today()
    for k in range(0, 75):
        d = (today - dt.timedelta(days=k)).strftime("%Y%m%d")
        for t in got:
            try:
                b = fetch(URL.format(t=t, d=d))
            except Exception as e:
                if k < 3:
                    print(f"::warning::{t} {d}: {e}")
                continue
            if len(b) < 50 or b.lstrip()[:1] in (b"<", b"{"):
                if k < 3:
                    print(f"::warning::{t} {d}: not CSV ({b[:120]!r})")
                continue
            open(f"{OUT}/{t}_{d}.csv", "wb").write(b)
            try:
                df = pd.read_csv(io.BytesIO(b)); df["file_date"] = d; got[t].append(df)
            except Exception as e:
                print(f"::warning::parse {t} {d}: {e}")
    for t, L in got.items():
        if not L:
            print(f"::warning::{t}: no files"); continue
        a = pd.concat(L, ignore_index=True)
        a.to_parquet(f"{OUT}/{t}_all.parquet", index=False)
        print(f"::notice::{t}: {len(L)} days, {len(a):,} rows, columns {list(a.columns)[:30]}")
        print(f"=== {t} head\n{a.head(8).to_string()}")
    try:                                            # index of the extra dataset host, if any
        b = fetch("https://data.forecastex.com/")
        open(f"{OUT}/data_host_index.html", "wb").write(b)
        print(f"::notice::data.forecastex.com index {len(b)} bytes: {b[:300]!r}")
    except Exception as e:
        print(f"::warning::data host: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        for line in traceback.format_exc().strip().splitlines()[-10:]:
            print(f"::error::{line}")
        raise
