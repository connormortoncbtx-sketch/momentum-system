"""research/fetch_asos1min.py -- 1-minute ASOS temperatures (IEM archive) for the Kalshi stations.

Why: the NWS daily climate report (what Kalshi settles on) takes its max from the 1-minute
data; METAR obs (fetch_asos.py) can miss the true max between reports and add C->F rounding.
Stored as the max per 5-minute bin to keep the file small.
Output: research_data/kalshi/asos1min.parquet  (station, valid_utc, tmpf)   [same schema as asos.parquet]
"""
import io, os, time, urllib.request
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = f"{RD}/kalshi"
STATIONS = ["NYC", "MDW", "MIA", "AUS", "LAX", "DEN", "PHL", "DCA", "BOS", "DFW", "PHX", "SAT", "SAN", "EWR"]
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"


def month(st, y, m):
    a = pd.Timestamp(y, m, 1); b = a + pd.offsets.MonthBegin(1)
    q = (f"station={st}&tz=UTC&year1={a.year}&month1={a.month}&day1=1&hour1=0&minute1=0"
         f"&year2={b.year}&month2={b.month}&day2=1&hour2=0&minute2=0&vars=tmpf&sample=1min&what=download&delim=comma")
    for k in range(4):
        try:
            txt = urllib.request.urlopen(urllib.request.Request(f"{URL}?{q}", headers={"User-Agent": "research"}),
                                         timeout=300).read().decode()
            d = pd.read_csv(io.StringIO(txt))
            if d.empty or "tmpf" not in d.columns:
                return pd.DataFrame()
            tcol = [c for c in d.columns if "utc" in c.lower() or c.lower() in ("valid", "valid(utc)")][0]
            d = d.rename(columns={tcol: "valid_utc"})
            d["valid_utc"] = pd.to_datetime(d.valid_utc, utc=True, errors="coerce")
            d["tmpf"] = pd.to_numeric(d.tmpf, errors="coerce")
            d = d.dropna(subset=["valid_utc", "tmpf"])
            # ASOS reports a 5-minute running AVERAGE of the 1-minute values (that is what the daily
            # max is built from); take the max of that average within each 5-minute bin
            x = d.set_index("valid_utc").tmpf.sort_index()
            x = x[~x.index.duplicated()].asfreq("1min")
            avg = x.rolling(5, min_periods=4).mean()
            d = avg.resample("5min").max().dropna().reset_index()
            d["station"] = st
            return d
        except Exception as e:
            print(f"  {st} {y}-{m:02d} try {k}: {str(e)[:120]}", flush=True)
            time.sleep(10 * (k + 1))
    return pd.DataFrame()


def main():
    os.makedirs(OUT, exist_ok=True)
    part = f"{OUT}/asos5avg_parts"; os.makedirs(part, exist_ok=True)
    # probe first: does the endpoint answer with the expected columns?
    p = month("NYC", 2025, 7)
    print(f"::notice::probe NYC 2025-07: {len(p)} 5-min rows, max {p.tmpf.max() if len(p) else None}")
    if p.empty:
        raise SystemExit("asos1min probe returned nothing")
    for st in STATIONS:
        for y in range(2021, 2027):
            for m in range(1, 13):
                if pd.Timestamp(y, m, 1) < pd.Timestamp("2021-06-01") or pd.Timestamp(y, m, 1) > pd.Timestamp("2026-10-01"):
                    continue
                f = f"{part}/{st}_{y}{m:02d}.parquet"
                if os.path.exists(f):
                    continue
                d = month(st, y, m)
                d.to_parquet(f, index=False)
        print(f"  {st} done", flush=True)
    import glob
    df = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{part}/*.parquet") if os.path.getsize(f) > 500], ignore_index=True)
    df["tmpf"] = df.tmpf.astype("float32")
    df[["station", "valid_utc", "tmpf"]].to_parquet(f"{OUT}/asos5avg.parquet", index=False)
    print(f"::notice::ASOS 5-min running average: {len(df):,} 5-min bins, {df.station.nunique()} stations, "
          f"{df.valid_utc.min():%Y-%m}..{df.valid_utc.max():%Y-%m}; per station {df.station.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
