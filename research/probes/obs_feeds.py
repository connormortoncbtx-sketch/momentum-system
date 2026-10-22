"""
research/probes/obs_feeds.py -- how fast and how precise are the real-time station-reading feeds? (Oct 22, 2026)

obs_decided.py found the largest edge in any Kalshi test: buckets already decided by the station's precise 5-minute
readings still traded ~75c (+22.6c/contract, 351 city-days 2025-26) -- but those readings were assembled after the
fact. The earlier NWS probe found the free api.weather.gov feed ~20 min late in whole degrees C. This probe polls
every feed reachable without an account, for the bot's 11 stations, for PROBE_MIN minutes (default 70), and records
for every observation: the time it describes, the time we first saw it, and its value/precision.
  awc      aviationweather.gov METAR API (hourly + specials; METAR T-group carries tenths of C)
  nws      api.weather.gov latest observation (includes 5-minute high-frequency obs at many ASOS)
  iem      Iowa Environmental Mesonet current-conditions JSON (LDM-fed)
  iem1min  IEM ASOS 1-minute archive: newest minute available (is it real time at all?)
Summary per feed: median / 90th pct delay (first seen - observation time), observations per hour, decimals seen.
Results: results/obs_feeds.csv (every first sighting) + annotation.
"""
import csv, datetime as dt, json, os, statistics, time, urllib.request

STATIONS = {"KNYC": ("NYC", "NY_ASOS"), "KMDW": ("MDW", "IL_ASOS"), "KLAX": ("LAX", "CA_ASOS"),
            "KMIA": ("MIA", "FL_ASOS"), "KPHL": ("PHL", "PA_ASOS"), "KDEN": ("DEN", "CO_ASOS"),
            "KAUS": ("AUS", "TX_ASOS"), "KDFW": ("DFW", "TX_ASOS"), "KATL": ("ATL", "GA_ASOS"),
            "KBOS": ("BOS", "MA_ASOS"), "KLAS": ("LAS", "NV_ASOS")}
UA = {"User-Agent": "momentum-research (github actions probe)", "Accept": "application/json"}
OUT = os.path.join(os.environ.get("RESEARCH_DATA", "research_data"), "results")


def get(url, timeout=20):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return json.loads(r.read())


def utc(s):
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return dt.datetime.fromtimestamp(s, dt.timezone.utc)
    s = str(s).replace("Z", "+00:00")
    d = dt.datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def decimals(v):
    if v is None:
        return None
    s = repr(float(v)).rstrip("0").rstrip(".")
    return len(s.split(".")[1]) if "." in s else 0


def poll_awc():
    d = get("https://aviationweather.gov/api/data/metar?format=json&ids=" + ",".join(STATIONS))
    return [("awc", x.get("icaoId"), utc(x.get("obsTime")), x.get("temp"), (x.get("rawOb") or "")[:120]) for x in d]


def poll_nws(st):
    d = get(f"https://api.weather.gov/stations/{st}/observations?limit=3")
    out = []
    for f in d.get("features", []):
        p = f.get("properties", {})
        out.append(("nws", st, utc(p.get("timestamp")), (p.get("temperature") or {}).get("value"), (p.get("rawMessage") or "")[:120]))
    return out


def poll_iem(st, sid, net):
    d = get(f"https://mesonet.agron.iastate.edu/json/current.py?station={sid}&network={net}")
    ob = d.get("last_ob") or {}
    return [("iem", st, utc(ob.get("utc_valid") or ob.get("local_valid")), ob.get("airtempF") or ob.get("tmpf"),
             (ob.get("raw") or "")[:120])]


def poll_iem1min(st, sid):
    now = dt.datetime.now(dt.timezone.utc)
    s = now - dt.timedelta(hours=3)
    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py?station=" + sid + "&vars=tmpf&sample=1min&what=view"
           f"&tz=UTC&sts={s:%Y-%m-%dT%H:%MZ}&ets={now:%Y-%m-%dT%H:%MZ}")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        lines = r.read().decode("utf-8", "replace").strip().splitlines()
    rows = [l.split(",") for l in lines[1:] if l.count(",") >= 2]
    if not rows:
        return []
    last = rows[-1]
    try:
        t = dt.datetime.strptime(last[1].strip(), "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)
    except Exception:
        return []
    return [("iem1min", st, t, last[-1].strip(), "")]


def main():
    minutes = float(os.environ.get("PROBE_MIN", "70"))
    seen, rows, errs = {}, [], {}
    t_end = time.time() + minutes * 60
    n = 0
    while time.time() < t_end:
        n += 1
        batch = []
        for name, fn in (("awc", poll_awc),):
            try:
                batch += fn()
            except Exception as e:
                errs[name] = str(e)[:150]
        for st, (sid, net) in STATIONS.items():
            for name, fn in (("nws", lambda: poll_nws(st)), ("iem", lambda: poll_iem(st, sid, net))):
                try:
                    batch += fn()
                except Exception as e:
                    errs[name] = str(e)[:150]
            if n % 10 == 1:                                    # 1-minute archive: every ~5 min is enough
                try:
                    batch += poll_iem1min(st, sid)
                except Exception as e:
                    errs["iem1min"] = str(e)[:150]
        now = dt.datetime.now(dt.timezone.utc)
        for src, st, t_obs, val, raw in batch:
            if t_obs is None:
                continue
            key = (src, st, t_obs.isoformat())
            if key in seen:
                continue
            seen[key] = now
            rows.append(dict(src=src, station=st, obs_time=t_obs.isoformat(), first_seen=now.isoformat(),
                             delay_s=(now - t_obs).total_seconds(), value=val, decimals=decimals(val) if val not in (None, "") else None,
                             raw=raw, first_poll=n == 1))
        time.sleep(max(0, 30 - (time.time() - now.timestamp())))
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/obs_feeds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["src"])
        w.writeheader(); w.writerows(rows)
    lines = []
    for src in ("awc", "nws", "iem", "iem1min"):
        r = [x for x in rows if x["src"] == src and not x["first_poll"]]          # first poll = backlog, not a delay
        allr = [x for x in rows if x["src"] == src]
        if not allr:
            lines.append(f"{src}: no data ({errs.get(src, 'no error')})"); continue
        if src == "iem1min":
            lag = statistics.median([x["delay_s"] for x in allr]) / 60
            lines.append(f"iem1min: newest minute is ~{lag:.0f} min old")
            continue
        d = sorted(x["delay_s"] / 60 for x in r)
        per_hr = len(r) / len(STATIONS) / (minutes / 60) if r else 0
        dec = sorted({x["decimals"] for x in allr if x["decimals"] is not None})
        lines.append(f"{src}: delay median {d[len(d) // 2]:.1f} min, p90 {d[int(len(d) * .9)]:.1f} min, {per_hr:.1f} obs/hr/station, "
                     f"decimals {dec}" if d else f"{src}: no new obs during the run, decimals {dec}")
    print("\n".join(lines))
    if errs:
        print("errors:", errs)
    print("::notice::obs feeds -- " + " | ".join(lines))


if __name__ == "__main__":
    main()
