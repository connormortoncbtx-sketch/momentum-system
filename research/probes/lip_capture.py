"""Probe: how much of Kalshi's Liquidity Incentive Program (LIP) could the bot earn on the weather markets?
For each active LIP weather market, snapshot the public order book every ~60 s for MINUTES minutes and apply the
program's scoring (help.kalshi.com, Liquidity Incentive Program): per side, walk down from the best bid; the
Reference Price is the first level where cumulative size >= Target/5; orders at/above it score full, below it are
discounted by discount_factor ^ ticks; only size that helps reach Target counts; our score share = our score /
(total qualifying score incl. ours); the period reward is split across the two sides and all snapshots.
Reports, for a hypothetical 5-contract order at the best bid / 1c / 2c behind on each side: expected reward per
contract per day, the capital it ties up (= the bid price), and reward per $ of capital per day."""
import json, os, time, urllib.parse, urllib.request
import numpy as np, pandas as pd
API = "https://api.elections.kalshi.com/trade-api/v2"
MINUTES = float(os.environ.get("MINUTES", "15")); OURS = 5.0
RD = os.environ.get("RESEARCH_DATA", "research_data")
def get(path, **q):
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{API}{path}?{urllib.parse.urlencode(q)}", headers={"Accept": "application/json"}), timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(1 + i)
    return {}
progs, cur = [], None
while True:
    q = dict(status="active", type="liquidity", limit=1000)
    if cur: q["cursor"] = cur
    d = get("/incentive_programs", **q); progs += d.get("incentive_programs", []) or []; cur = d.get("next_cursor")
    if not cur: break
P = pd.DataFrame(progs)
P = P[P.market_ticker.str.match(r"^KXHIGH(NY|CHI|LAX|MIA|PHIL|DEN|AUS|TDAL|TATL|TBOS|TLV)-")].copy()
for c in ("start_date", "end_date"): P[c] = pd.to_datetime(P[c], utc=True, format="ISO8601")
P["side_usd_per_sec"] = P.period_reward / 1e4 / 2 / (P.end_date - P.start_date).dt.total_seconds()
print(f"::notice::LIP weather markets probed: {len(P)}")

def score_side(levels, target, disc, our_px):
    """levels: list of (price, size) best first. Returns (our_score, total_score) for OURS contracts at our_px."""
    lv = sorted([(float(p), float(s)) for p, s in levels], key=lambda x: -x[0])
    lv = [(p, s + (OURS if abs(p - our_px) < 1e-9 else 0.0)) for p, s in lv]
    if not any(abs(p - our_px) < 1e-9 for p, _ in lv):
        lv = sorted(lv + [(our_px, OURS)], key=lambda x: -x[0])
    cum, ref = 0.0, None
    for p, s in lv:
        cum += s
        if cum >= target / 5:
            ref = p; break
    if ref is None or cum < 0:
        return 0.0, 0.0
    total, ours, cum = 0.0, 0.0, 0.0
    for p, s in lv:
        take = min(s, max(0.0, target - cum)); cum += s
        if take <= 0: break
        ticks = max(0, round((ref - p) * 100))
        w = (disc ** ticks)
        total += take * w
        if abs(p - our_px) < 1e-9:
            ours += min(OURS, take) * w
    return ours, total

rows = []
t_end = time.time() + MINUTES * 60
while time.time() < t_end:
    for r in P.itertuples():
        ob = get(f"/markets/{r.market_ticker}/orderbook").get("orderbook_fp") or {}
        disc = (r.discount_factor_bps or 5000) / 1e4; tgt = float(r.target_size_fp or 1000)
        for side in ("yes", "no"):
            lv = ob.get(f"{side}_dollars") or []
            other = ob.get(("no" if side == "yes" else "yes") + "_dollars") or []
            if not lv or not other:
                continue
            tot_depth = sum(float(s) for _, s in lv)
            best = max(float(p) for p, _ in lv)
            for k in (0, 1, 2):
                px = round(best - k / 100, 2)
                if px < 0.01: continue
                o, t = score_side(lv, tgt, disc, px)
                share = o / t if t > 0 else 0.0
                valid = tot_depth >= tgt and sum(float(s) for _, s in other) >= tgt     # snapshot counts only if both sides >= target
                usd_day = share * r.side_usd_per_sec * 86400 * (1.0 if valid else 0.0)
                rows.append(dict(market=r.market_ticker, side=side, ticks_behind=k, price=px, depth=tot_depth, valid=valid,
                                 share=share, usd_day_order=usd_day, usd_day_per_contract=usd_day / OURS, cap=px * OURS))
    time.sleep(20)
D = pd.DataFrame(rows)
os.makedirs(f"{RD}/kalshi", exist_ok=True); D.to_csv(f"{RD}/kalshi/lip_capture.csv", index=False)
D["pb"] = pd.cut(D.price, [0, .1, .3, .5, .7, .9, 1])
g = D.groupby(["ticks_behind"]).agg(valid_pct=("valid", "mean"), usd_day_per_contract=("usd_day_per_contract", "mean"),
                                     cap_per_contract=("price", "mean"))
g["ret_per_day_pct"] = g.usd_day_per_contract / g.cap_per_contract * 100
for k, r in g.iterrows():
    print(f"::notice::{k} ticks behind best: snapshots valid {r.valid_pct*100:.0f}%, reward {r.usd_day_per_contract*100:.2f}c per contract-day, "
          f"capital {r.cap_per_contract*100:.0f}c/contract -> {r.ret_per_day_pct:.2f}%/day on capital")
h = D[D.ticks_behind == 0].groupby("pb", observed=True).agg(c_day=("usd_day_per_contract", "mean"), cap=("price", "mean"))
print("::notice::at best bid by price: " + "; ".join(f"{i}: {r.c_day*100:.2f}c/day on {r.cap*100:.0f}c" for i, r in h.iterrows()))
mk = D[D.ticks_behind == 0].groupby(["market", "side"]).usd_day_order.mean()
print(f"::notice::5-contract order at best bid on BOTH sides of all {D.market.nunique()} markets: ~${mk.sum():.2f}/day total, "
      f"capital ~${(D[D.ticks_behind == 0].groupby(['market','side']).price.mean() * OURS).sum():.0f}")
