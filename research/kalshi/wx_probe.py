"""
research/kalshi/wx_probe.py -- what hourly forecast data can we actually get? (Nov 9, 2026; Phase 1 of the fair-value
project). Probes, for one station-day:
 A. IEM text MOS archive: which models answer (NBS, NBE, NBH, NBX, LAV, GFS), their run times and columns (looking
    for hourly NBM runs and any max-temp spread / percentile columns)
 B. AWS NOAA open data: HRRR (hourly runs, 2 m TMP) and NBM (core + qmd percentile products) -- .idx listings for one
    run, the fields relevant to daily max temperature, and one byte-range download decoded with pygrib to a station
    value (KNYC Central Park) to prove the pipeline works.
Public data only. Prints findings.
"""
import io, json, os, re, sys, urllib.request, datetime as dt

DAY = dt.date(2026, 11, 3)
UA = {"User-Agent": "momentum-research (weather study)"}


def get(url, rng=None, timeout=60):
    h = dict(UA)
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
        return r.read()


def iem():
    print("== A. IEM MOS archive (KNYC)")
    for model in ("NBS", "NBE", "NBH", "NBX", "LAV", "GFS"):
        q = f"station=KNYC&model={model}&sts={DAY}T00:00Z&ets={DAY}T23:59Z&format=csv"
        try:
            t = get(f"https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?{q}").decode()
            lines = t.strip().splitlines()
            runs = sorted({l.split(",")[2] for l in lines[1:] if l.count(",") > 3})
            print(f"  {model}: {len(lines) - 1} rows, {len(runs)} runs {runs[:6]}{'...' if len(runs) > 6 else ''}")
            print(f"     cols: {lines[0][:300] if lines else ''}")
            if len(lines) > 1:
                print(f"     row : {lines[1][:300]}")
        except Exception as e:
            print(f"  {model}: {type(e).__name__} {str(e)[:120]}")


def idx_list(url):
    t = get(url + ".idx").decode()
    rows = []
    for l in t.strip().splitlines():
        p = l.split(":")
        rows.append((int(p[1]), ":".join(p[3:])))
    return rows


def aws():
    print("== B. AWS open data")
    d = DAY.strftime("%Y%m%d")
    cands = {
        "HRRR sfc f06": f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{d}/conus/hrrr.t12z.wrfsfcf06.grib2",
        "NBM core f12": f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{d}/13/core/blend.t13z.core.f012.co.grib2",
        "NBM qmd f18": f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{d}/13/qmd/blend.t13z.qmd.f018.co.grib2",
    }
    found = {}
    for name, url in cands.items():
        try:
            rows = idx_list(url)
            hits = [r for r in rows if re.search(r"TMP:2 m|TMAX|MAXT|TMP:2 m above ground:.*max", r[1], re.I)]
            print(f"  {name}: {len(rows)} fields; temperature-max related: {len(hits)}")
            for r in hits[:12]:
                print(f"     {r[1][:140]}")
            found[name] = (url, rows)
        except Exception as e:
            print(f"  {name}: {type(e).__name__} {str(e)[:150]}")
    # decode one field
    try:
        import pygrib, numpy as np
    except Exception as e:
        print(f"  pygrib unavailable: {e}"); return
    for name in ("HRRR sfc f06", "NBM core f12", "NBM qmd f18"):
        if name not in found:
            continue
        url, rows = found[name]
        for i, (off, desc) in enumerate(rows):
            if re.search(r"TMP:2 m above ground|TMAX:2 m above ground", desc):
                end = rows[i + 1][0] - 1 if i + 1 < len(rows) else off + 5_000_000
                b = get(url, (off, end))
                g = pygrib.fromstring(b)
                lats, lons = g.latlons()
                j = np.unravel_index(np.argmin((lats - 40.7789) ** 2 + (lons + 73.9692) ** 2), lats.shape)
                v = g.values[j]
                print(f"  decoded {name} '{desc[:70]}': {len(b)/1e6:.1f} MB, grid {lats.shape}, KNYC {v:.2f} K = {(v - 273.15) * 1.8 + 32:.1f} F")
                break


if __name__ == "__main__":
    iem(); aws()
