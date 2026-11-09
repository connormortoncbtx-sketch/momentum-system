"""
research/kalshi/nbm_archive.py -- Phase 1 of the fair-value project (Nov 9, 2026): the National Blend of Models'
HOURLY gridded daytime-max forecast for each settlement station, from NOAA's open archive on AWS.

Why: the Oct 8 / Oct 22 studies used the 4x-daily NBM TEXT product, and found the market sharper than it and not
lagging it. The sharp traders' input is the hourly gridded NBM (and HRRR). Each hourly NBM run carries
TMAX (12Z-00Z window = the US daytime max) and its ensemble standard deviation -- a probabilistic forecast of the
day's high. This collects it for every hourly run that covers each target day.

For target day D (the daytime window ending D+1 00Z) and every run R from D-1 00Z to D 12Z: file
blend.tRRz.core.fFFF.co.grib2 with FFF = hours from R to D+1 00Z; the two GRIB messages
'TMAX:2 m above ground:a-FFF hour max fcst:' and the same ':ens std dev' are fetched by byte range (.idx) and
read at the grid point nearest each station. Runs are public about 1h after their nominal time (applied later).
Output: research_data/kalshi/nbm/<D>.parquet (resumable) and research_data/kalshi/nbm_tmax.parquet (all):
  day, run (UTC), lead_h, station, series, mean_f, sd_f
Env: DAYS (default 150) back from yesterday. Public data only.
"""
import concurrent.futures as cf
import datetime as dt
import os
import re
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = f"{RD}/kalshi/nbm"; os.makedirs(OUT, exist_ok=True)
BASE = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
UA = {"User-Agent": "momentum-research (weather study)"}
DAYS = int(os.environ.get("DAYS") or os.environ.get("LIMIT") or 150)
# settlement stations (kalshi_mm/sniper.py) with coordinates
ST = {"KXHIGHNY": ("KNYC", 40.7789, -73.9692), "KXHIGHCHI": ("KMDW", 41.7868, -87.7522),
      "KXHIGHLAX": ("KLAX", 33.9382, -118.3866), "KXHIGHMIA": ("KMIA", 25.7881, -80.3169),
      "KXHIGHPHIL": ("KPHL", 39.8733, -75.2268), "KXHIGHDEN": ("KDEN", 39.8466, -104.6562),
      "KXHIGHAUS": ("KAUS", 30.1831, -97.6799), "KXHIGHTDAL": ("KDFW", 32.8968, -97.0380),
      "KXHIGHTATL": ("KATL", 33.6301, -84.4418), "KXHIGHTBOS": ("KBOS", 42.3606, -71.0097),
      "KXHIGHTLV": ("KLAS", 36.0719, -115.1634), "KXHIGHTHOU": ("KHOU", 29.6375, -95.2825),
      "KXHIGHTMIN": ("KMSP", 44.8831, -93.2289), "KXHIGHTPHX": ("KPHX", 33.4278, -112.0037),
      "KXHIGHTSEA": ("KSEA", 47.4447, -122.3144), "KXHIGHTSFO": ("KSFO", 37.6197, -122.3656)}
_IDX = None


def get(url, rng=None, tries=4):
    h = dict(UA)
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=90) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 + 3 * i)
        except Exception:
            time.sleep(2 + 3 * i)
    return None


def fetch(run, f):
    """(run, f, mean bytes, sd bytes) or None."""
    url = f"{BASE}/blend.{run:%Y%m%d}/{run:%H}/core/blend.t{run:%H}z.core.f{f:03d}.co.grib2"
    idx = get(url + ".idx")
    if not idx:
        return None
    lines = idx.decode().strip().splitlines()
    offs = [int(l.split(":")[1]) for l in lines]
    want = {}
    for i, l in enumerate(lines):
        m = re.search(rf"TMAX:2 m above ground:\d+-{f} hour max fcst:(ens std dev)?", l)
        if m:
            end = offs[i + 1] - 1 if i + 1 < len(offs) else offs[i] + 8_000_000
            want["sd" if m.group(1) else "mean"] = (offs[i], end)
    if "mean" not in want:
        return None
    mb = get(url, want["mean"]); sb = get(url, want["sd"]) if "sd" in want else None
    return (run, f, mb, sb) if mb else None


def decode(item):
    import pygrib
    global _IDX
    run, f, mb, sb = item
    g = pygrib.fromstring(mb)
    if _IDX is None:
        lats, lons = g.latlons()
        _IDX = {s: np.unravel_index(np.argmin((lats - la) ** 2 + ((lons - lo + 180) % 360 - 180) ** 2), lats.shape)
                for s, (_, la, lo) in ST.items()}
    v = g.values
    sd = pygrib.fromstring(sb).values if sb else None
    k = lambda x: (x - 273.15) * 1.8 + 32
    return [(run, f, s, ST[s][0], float(k(v[j])), float(sd[j] * 1.8) if sd is not None else np.nan) for s, j in _IDX.items()]


def day_runs(D):
    end = dt.datetime(D.year, D.month, D.day, tzinfo=dt.timezone.utc) + dt.timedelta(days=1)   # D+1 00Z
    out = []
    for h in range(36, 11, -1):                                  # runs D-1 12Z .. D 12Z
        run = end - dt.timedelta(hours=h)
        out.append((run, h))
    return out


def main():
    today = dt.datetime.now(dt.timezone.utc).date()
    days = [today - dt.timedelta(days=i) for i in range(1, DAYS + 1)]
    todo = [D for D in days if not os.path.exists(f"{OUT}/{D}.parquet")]
    print(f"{len(days)} target days, {len(todo)} to fetch", flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(12) as io_ex, cf.ProcessPoolExecutor(4) as cpu:
        for n, D in enumerate(todo):
            items = [x for x in io_ex.map(lambda rf: fetch(*rf), day_runs(D)) if x]
            rows = [r for rs in cpu.map(decode, items) for r in rs]
            df = pd.DataFrame(rows, columns=["run", "lead_h", "series", "station", "mean_f", "sd_f"])
            df["day"] = str(D)
            df.to_parquet(f"{OUT}/{D}.parquet", index=False)
            if n % 10 == 9 or n == len(todo) - 1:
                print(f"  {n + 1}/{len(todo)} days ({len(items)} runs on {D}); {time.time() - t0:.0f}s", flush=True)
    allf = [pd.read_parquet(f"{OUT}/{f}") for f in sorted(os.listdir(OUT)) if f.endswith(".parquet")]
    A = pd.concat(allf, ignore_index=True)
    A.to_parquet(f"{RD}/kalshi/nbm_tmax.parquet", index=False)
    per = A.groupby("day").run.nunique()
    print(f"::notice::nbm archive: {A.day.nunique()} days, {len(A):,} rows; runs per day median {per.median():.0f} "
          f"(min {per.min()}); sd median {A.sd_f.median():.1f}F")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
