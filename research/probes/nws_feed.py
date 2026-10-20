"""Probe: is the free NWS real-time observation feed (api.weather.gov) fast and precise enough to call Kalshi
daily-high buckets early? Polls latest observations for several Kalshi stations every 30 s for MINUTES minutes and
records, per new observation: its timestamp, when we first saw it (latency), temperature value and precision, and
the raw METAR text (hourly/special vs 5-minute 'high-frequency' reports)."""
import json, os, statistics, sys, time, traceback, urllib.request, datetime as dt


def _hook(t, v, tb):                                   # surface any crash as annotations (logs aren't readable)
    for line in "".join(traceback.format_exception(t, v, tb)).strip().splitlines()[-8:]:
        print(f"::error::{line}")
    sys.__excepthook__(t, v, tb)


sys.excepthook = _hook
import pandas as pd

STATIONS = os.environ.get("STATIONS", "KNYC,KMDW,KAUS,KMIA,KPHL,KDEN,KLAX,KDFW").split(",")
MINUTES = float(os.environ.get("MINUTES", "40"))
UA = {"User-Agent": "MomentumAlpha research (github.com/connormortoncbtx-sketch)", "Accept": "application/geo+json"}
RD = os.environ.get("RESEARCH_DATA", "research_data")


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
        return json.loads(r.read())


seen, rows, errs = set(), [], {}
t_end = time.time() + MINUTES * 60
first = True
while time.time() < t_end:
    now = dt.datetime.now(dt.timezone.utc)
    for st in STATIONS:
        try:
            feats = get(f"https://api.weather.gov/stations/{st}/observations?limit=12")["features"]
        except Exception as e:
            errs[str(e)[:120]] = errs.get(str(e)[:120], 0) + 1
            continue
        for f in feats:
            p = f["properties"]; ts = p["timestamp"]
            if (st, ts) in seen:
                continue
            seen.add((st, ts))
            v = (p.get("temperature") or {}).get("value")
            rows.append(dict(station=st, obs_time=ts, first_seen=now.isoformat(), backlog=first, temp_c=v,
                             raw=(p.get("rawMessage") or "")[:80], qc=(p.get("temperature") or {}).get("qualityControl")))
    first = False
    time.sleep(30)

for e, n in list(errs.items())[:5]:
    print(f"::warning::request errors x{n}: {e}")
if not rows:
    print("::error::no observations fetched at all"); raise SystemExit(1)
d = pd.DataFrame(rows)
d["obs_time"] = pd.to_datetime(d.obs_time, utc=True); d["first_seen"] = pd.to_datetime(d.first_seen, utc=True)
d["latency_min"] = (d.first_seen - d.obs_time).dt.total_seconds() / 60
d["tenths"] = d.temp_c.map(lambda v: v == v and v is not None and abs(round(v * 10) - round(v) * 10) > 0)
d["kind"] = d.raw.map(lambda r: "SPECI" if " SPECI " in f" {r} " or r.startswith("SPECI") else ("METAR" if r else "no raw text"))
os.makedirs(f"{RD}/kalshi", exist_ok=True)
d.to_csv(f"{RD}/kalshi/nws_feed_probe.csv", index=False)
live = d[~d.backlog]
for st, g in d.sort_values("obs_time").groupby("station"):
    gaps = g.obs_time.diff().dt.total_seconds().div(60).dropna()
    print(f"::notice::{st}: {len(g)} obs; spacing median {gaps.median() if len(gaps) else float('nan'):.0f} min; "
          f"temp with tenths {g.tenths.mean() * 100:.0f}%; kinds {g.kind.value_counts().to_dict()}; "
          f"live latency median {live[live.station == st].latency_min.median():.1f} min (n {len(live[live.station == st])}); "
          f"sample {g.temp_c.dropna().tail(4).tolist()} raw '{g.raw.iloc[-1][:40]}'")
print(f"::notice::ALL: live new obs {len(live)}, latency median {live.latency_min.median():.1f} min, p90 {live.latency_min.quantile(.9):.1f}; "
      f"obs spacing median {d.sort_values('obs_time').groupby('station').obs_time.diff().dt.total_seconds().div(60).median():.0f} min; "
      f"tenths {d.tenths.mean() * 100:.0f}%")
