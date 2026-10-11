"""research/fetch_asos.py -- hourly/5-min ASOS temperature observations (IEM archive) for the
Kalshi daily-high stations, June 2021 on, for the same-day "max so far" study.
Output: research_data/kalshi/asos.parquet  (station, valid_utc, tmpf)"""
import io, os, time, urllib.request
from pathlib import Path
import pandas as pd
OUT = Path(os.environ.get("RESEARCH_DATA", "research_data")) / "kalshi"
UA = os.environ.get("SEC_USER_AGENT") or "MomentumAlpha research"
STATIONS = ["NYC", "MDW", "MIA", "AUS", "LAX", "DEN", "PHL", "DCA", "BOS", "DFW", "PHX", "SAT", "SAN", "EWR"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for st in STATIONS:
        for yr in range(2021, pd.Timestamp.today().year + 1):
            q = (f"station={st}&data=tmpf&year1={yr}&month1=1&day1=1&year2={yr + 1}&month2=1&day2=1"
                 f"&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=3&report_type=4")
            for i in range(5):
                try:
                    req = urllib.request.Request(f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?{q}",
                                                 headers={"User-Agent": UA})
                    txt = urllib.request.urlopen(req, timeout=300).read().decode()
                    break
                except Exception:
                    time.sleep(15 * (i + 1)); txt = ""
            d = pd.read_csv(io.StringIO(txt), na_values=["M"]) if txt else pd.DataFrame()
            if len(d):
                d = d.rename(columns={"station": "station", "valid": "valid_utc"}).dropna(subset=["tmpf"])
                d["station"] = "K" + st
                frames.append(d[["station", "valid_utc", "tmpf"]])
            time.sleep(2)
    df = pd.concat(frames)
    df["valid_utc"] = pd.to_datetime(df.valid_utc, utc=True)
    df["tmpf"] = df.tmpf.astype("float32")
    df.to_parquet(OUT / "asos.parquet", index=False)
    print(f"::notice::ASOS: {len(df):,} obs, {df.station.nunique()} stations, {df.valid_utc.min():%Y-%m}..{df.valid_utc.max():%Y-%m}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-10:]:
            print(f"::error::{line}")
        raise
