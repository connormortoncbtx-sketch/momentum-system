"""Probe: what does Polymarket US list (public gateway), how liquid/wide is it, what do its liquidity incentives pay,
and do any of the same events trade on Kalshi at different prices (cross-venue arbitrage check)?"""
import json, os, re, time, urllib.parse, urllib.request, collections
import pandas as pd
G = "https://gateway.polymarket.us"
K = "https://api.elections.kalshi.com/trade-api/v2"
RD = os.environ.get("RESEARCH_DATA", "research_data")
def get(base, path, **q):
    url = f"{base}{path}" + (("?" + urllib.parse.urlencode(q, doseq=True)) if q else "")
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"}), timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            err = e; time.sleep(1 + i)
    print(f"::warning::{url[:100]}: {err}"); return {}
# ---- markets
mk, off = [], 0
while off < 20000:
    d = get(G, "/v1/markets", limit=500, offset=off, active="true", closed="false")
    items = d.get("markets", d if isinstance(d, list) else [])
    if not items: break
    mk += items; off += len(items)
    if len(items) < 500: break
print(f"::notice::PM-US active markets: {len(mk)}; keys: {sorted(mk[0].keys())[:45] if mk else None}")
M = pd.json_normalize(mk) if mk else pd.DataFrame()
os.makedirs(f"{RD}/pmus", exist_ok=True); M.to_parquet(f"{RD}/pmus/markets.parquet") if len(M) else None
for col in ("category", "categorySlug", "subjectSlug", "tagSlug", "sportsMarketType", "marketType"):
    if col in M:
        print(f"::notice::by {col}: {M[col].astype(str).value_counts().head(12).to_dict()}")
for col in ("volume", "volumeNum", "volume24hr", "liquidity", "spread"):
    if col in M:
        print(f"::notice::{col} median {pd.to_numeric(M[col], errors='coerce').median()}, sum {pd.to_numeric(M[col], errors='coerce').sum():,.0f}")
txt = M.get("question", M.get("title", pd.Series(dtype=str))).astype(str)
for kw in ("temperature", "high temp", "weather", "rain", "CPI", "Fed", "gas", "oil", "Bitcoin", "election"):
    print(f"::notice::markets mentioning '{kw}': {int(txt.str.contains(kw, case=False).sum())}")
print(f"::notice::sample titles: {txt.sample(min(8, len(txt)), random_state=1).tolist() if len(txt) else []}")
# ---- book sample
if len(M) and "slug" in M:
    samp = M.sample(min(40, len(M)), random_state=2).slug.tolist()
    rows = []
    for s in samp:
        b = get(G, f"/v1/markets/{s}/bbo")
        rows.append({"slug": s, **{k: v for k, v in (b.get("marketData") or b).items() if not isinstance(v, (dict, list))}})
    B = pd.DataFrame(rows); B.to_csv(f"{RD}/pmus/bbo_sample.csv", index=False)
    print(f"::notice::BBO keys {list(B.columns)[:30]}")
    print(f"::notice::BBO sample {B.head(3).to_dict('records')}")
# ---- incentives
inc = get(G, "/v1/incentives", limit=500)
progs = inc.get("incentivePrograms", inc.get("programs", inc if isinstance(inc, list) else []))
print(f"::notice::PM-US incentive programs: {len(progs)}; sample {json.dumps(progs[:2])[:700] if progs else inc and str(inc)[:300]}")
if progs:
    rows = []
    for p in progs:
        for tp in p.get("timePeriods", []):
            rows.append(dict(market=p.get("marketSlug"), start=tp.get("startTime") or tp.get("start"), end=tp.get("endTime") or tp.get("end"),
                             pool=tp.get("rewardPool"), target=tp.get("targetSize"), disc=tp.get("discountFactor"), maxspread=tp.get("maxSpread")))
    I = pd.DataFrame(rows); I.to_csv(f"{RD}/pmus/incentives.csv", index=False)
    print(f"::notice::incentive periods {len(I)}; pool stats {pd.to_numeric(I.pool, errors='coerce').describe().round(1).to_dict()}")
