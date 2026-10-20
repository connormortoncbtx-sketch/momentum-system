"""Probe: Kalshi liquidity incentive programs (GET /incentive_programs): which markets pay, how much, target sizes,
discount factors -- and whether any of the bot's weather series or the shortlist are covered."""
import json, urllib.request, urllib.parse, collections
import pandas as pd
API = "https://api.elections.kalshi.com/trade-api/v2"
def get(path, **q):
    with urllib.request.urlopen(urllib.request.Request(f"{API}{path}?{urllib.parse.urlencode(q)}", headers={"Accept": "application/json"}), timeout=60) as r:
        return json.loads(r.read())
rows, cur = [], None
for _ in range(50):
    q = dict(status="active", type="liquidity", limit=1000)
    if cur: q["cursor"] = cur
    d = get("/incentive_programs", **q)
    rows += d.get("incentive_programs", []) or []
    cur = d.get("next_cursor")
    if not cur: break
d = pd.DataFrame(rows)
print(f"::notice::active liquidity programs: {len(d)}; columns {list(d.columns)}")
if len(d):
    d["series"] = d.market_ticker.str.split("-").str[0]
    d["days"] = (pd.to_datetime(d.end_date) - pd.to_datetime(d.start_date)).dt.total_seconds() / 86400
    d["reward_usd"] = d.period_reward / 10000.0                      # centi-cents -> dollars
    d["usd_per_day"] = d.reward_usd / d.days.clip(lower=1 / 24)
    d["target"] = pd.to_numeric(d.get("target_size_fp"), errors="coerce")
    g = d.groupby("series").agg(markets=("market_ticker", "size"), usd_per_day=("usd_per_day", "sum"),
                                med_target=("target", "median"), disc_bps=("discount_factor_bps", "median"),
                                desc=("incentive_description", "first")).sort_values("usd_per_day", ascending=False)
    print(f"::notice::total paid per day across active programs ~${d.usd_per_day.sum():,.0f}; series {len(g)}")
    for s, r in g.head(25).iterrows():
        print(f"::notice::{s}: {r.markets} mkts, ${r.usd_per_day:,.1f}/day, target {r.med_target}, discount {r.disc_bps}bps | {str(r.desc)[:60]}")
    mine = [s for s in g.index if s.startswith(("KXHIGH", "KXLOW", "KXWTI", "KXBTCMAX", "KXSPACEX", "KXCPI", "KXRAIN"))]
    print(f"::notice::bot-relevant series with programs: {mine[:30]}")
    for s in mine[:15]:
        r = g.loc[s]; print(f"::notice::  {s}: {r.markets} mkts ${r.usd_per_day:,.2f}/day target {r.med_target} discount {r.disc_bps}bps")
    print(d.head(3).to_dict("records"))
