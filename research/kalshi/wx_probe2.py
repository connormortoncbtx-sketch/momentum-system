"""wx_probe2: where does NBM put daily MAX temperature (and its spread/percentiles) on AWS, and how often? +
IEM NBS run times (txn/xnd). Nov 9, 2026."""
import re, time, urllib.request, datetime as dt
UA = {"User-Agent": "momentum-research (weather study)"}
D = "20261103"


def get(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60) as r:
        return r.read().decode()


def ls(prefix):
    x = get(f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/?list-type=2&prefix={prefix}&delimiter=/")
    return re.findall(r"<Prefix>([^<]+)</Prefix>", x)[1:], re.findall(r"<Key>([^<]+)</Key>", x)


print("run dirs:", ls(f"blend.{D}/")[0][:30])
subs, _ = ls(f"blend.{D}/13/")
print("13z products:", subs)
for prod in subs:
    _, keys = ls(prod)
    g = [k for k in keys if k.endswith(".grib2")]
    print(f"  {prod}: {len(g)} files, e.g. {g[:3]}")
_, keys = ls(f"blend.{D}/13/core/")
for k in sorted(k for k in keys if k.endswith(".co.grib2"))[:40]:
    try:
        idx = get(f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/{k}.idx")
    except Exception as e:
        continue
    hits = [l.split(":", 3)[3] for l in idx.splitlines() if re.search(r"TMAX|TMIN|MAXT", l)]
    if hits:
        print(f"  {k.split('/')[-1]}: " + " | ".join(h[:90] for h in hits[:8]))
for prod in [p for p in subs if "qmd" in p]:
    _, keys = ls(prod)
    for k in sorted(k for k in keys if k.endswith(".grib2"))[:30]:
        idx = get(f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/{k}.idx")
        hits = [l.split(":", 3)[3] for l in idx.splitlines() if re.search(r"TMAX|MAXT", l)]
        if hits:
            print(f"  QMD {k.split('/')[-1]}: {len(hits)} fields, e.g. " + " | ".join(h[:80] for h in hits[:6]))
            break
time.sleep(5)
t = get(f"https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?station=KNYC&model=NBS&sts=2026-11-02T00:00Z&ets=2026-11-03T23:59Z&format=csv")
rows = [l.split(",") for l in t.strip().splitlines()]
h = rows[0]; ri = h.index("runtime"); ti = h.index("txn"); xi = h.index("xnd"); fi = h.index("ftime")
runs = sorted({r[ri] for r in rows[1:]})
print("IEM NBS runs Nov 2-3:", runs)
print("txn/xnd examples:", [(r[ri][11:16], r[fi][5:16], r[ti], r[xi]) for r in rows[1:] if r[ti]][:12])
