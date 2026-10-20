"""Probe: cross-venue arbitrage between Polymarket US and Kalshi on the same game-winner markets.
Match: PM slug aec-{lg}-{a}-{b}-{date} <-> Kalshi KX{LG}GAME event with the same date and team codes.
Orientation: PM bestBid/bestAsk are assumed to quote outcomes[0] (= first team in the slug); checked by comparing
mids against Kalshi (if the assumption were wrong, PM mid would track 1 - Kalshi mid).
Arb (2-way, no draws): buy A on one venue at its ask and B on the other (B on PM = selling A at PM bid):
  profit/contract = bid_other(A) - ask_this(A) - taker fees (Kalshi 0.07 p(1-p), PM 0.0695 p(1-p)) per leg.
Snapshots every 60 s for MINUTES minutes."""
import json, os, re, time, urllib.parse, urllib.request
import pandas as pd
G, K = "https://gateway.polymarket.us", "https://api.elections.kalshi.com/trade-api/v2"
MINUTES = float(os.environ.get("MINUTES", "20")); RD = os.environ.get("RESEARCH_DATA", "research_data")
def get(base, path, **q):
    url = f"{base}{path}" + (("?" + urllib.parse.urlencode(q)) if q else "")
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"}), timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(1 + i)
    return {}
SER = {"nba": "KXNBAGAME", "nfl": "KXNFLGAME", "cfb": "KXNCAAFGAME"}
MON = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6, JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)
def pm_games():
    out, off = [], 0
    while off < 20000:
        d = get(G, "/v1/markets", limit=500, offset=off, active="true", closed="false")
        it = d.get("markets", []); off += len(it)
        out += [m for m in it if m.get("marketType") == "moneyline" and re.match(r"aec-(nba|nfl|cfb)-", m.get("slug", ""))]
        if len(it) < 500: break
    return out
def kalshi_games():
    out = {}
    for lg, s in SER.items():
        for m in (get(K, "/markets", series_ticker=s, status="open", limit=1000).get("markets") or []):
            ev = m.get("event_ticker", ""); tm = m["ticker"].split("-")[-1]
            mm = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{4})?([A-Z]+)$", ev)
            if not mm: continue
            date = f"20{mm.group(1)}-{MON[mm.group(2)]:02d}-{int(mm.group(3)):02d}"
            out[(lg, date, tm)] = dict(bid=float(m.get("yes_bid_dollars") or 0), ask=float(m.get("yes_ask_dollars") or 0),
                                       ev=ev, teams=mm.group(5), vol=float(m.get("volume_24h_fp") or 0))
    return out
fee_k = lambda p: 0.07 * p * (1 - p); fee_p = lambda p: 0.0695 * p * (1 - p)
rows = []
t_end = time.time() + MINUTES * 60
while time.time() < t_end:
    pm, kg = pm_games(), kalshi_games()
    now = pd.Timestamp.now(tz="UTC")
    for m in pm:
        p = m["slug"].split("-"); lg, a, b = p[1], p[2].upper(), p[3].upper(); date = "-".join(p[4:7])
        ka = kg.get((lg, date, a)) or kg.get((lg, (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), a))
        bid = (m.get("bestBidQuote") or {}).get("value"); ask = (m.get("bestAskQuote") or {}).get("value")
        if not ka or bid is None or ask is None or ka["ask"] <= 0:
            continue
        pb, pa = float(bid), float(ask)
        arb1 = pb - ka["ask"] - fee_k(ka["ask"]) - fee_p(pb)          # buy A on Kalshi, sell A (buy B) on PM
        arb2 = ka["bid"] - pa - fee_p(pa) - fee_k(ka["bid"])          # buy A on PM, sell A (buy B) on Kalshi
        rows.append(dict(t=now, lg=lg, slug=m["slug"], team=a, pm_bid=pb, pm_ask=pa, k_bid=ka["bid"], k_ask=ka["ask"],
                         pm_mid=(pb + pa) / 2, k_mid=(ka["bid"] + ka["ask"]) / 2, arb_buyK_sellPM=arb1, arb_buyPM_sellK=arb2,
                         k_vol24=ka["vol"]))
    time.sleep(60)
D = pd.DataFrame(rows)
os.makedirs(f"{RD}/pmus", exist_ok=True); D.to_csv(f"{RD}/pmus/xvenue_arb.csv", index=False)
if not len(D):
    print("::notice::no matched games"); raise SystemExit
D["best"] = D[["arb_buyK_sellPM", "arb_buyPM_sellK"]].max(axis=1)
g = D.groupby("slug").agg(lg=("lg", "first"), n=("t", "size"), mid_gap=("pm_mid", lambda v: 0), best=("best", "max"))
corr = D[["pm_mid", "k_mid"]].corr().iloc[0, 1]
gap = (D.pm_mid - D.k_mid).abs()
print(f"::notice::matched games {D.slug.nunique()} ({D.lg.value_counts().to_dict()}), snapshots {len(D)}; PM vs Kalshi mid corr {corr:.3f} "
      f"(orientation ok if ~+1); median |mid gap| {gap.median()*100:.2f}c, p90 {gap.quantile(.9)*100:.2f}c")
print(f"::notice::best net arb per contract (after both taker fees): median {D.best.median()*100:.2f}c, max {D.best.max()*100:.2f}c; "
      f"snapshots with arb > 0: {(D.best > 0).mean()*100:.1f}%; > 1c: {(D.best > 0.01).mean()*100:.1f}%; games ever > 0: {(g.best > 0).sum()}")
top = D.sort_values("best", ascending=False).head(5)
for r in top.itertuples():
    print(f"::notice::  {r.slug} {r.team}: PM {r.pm_bid:.3f}/{r.pm_ask:.3f}  Kalshi {r.k_bid:.2f}/{r.k_ask:.2f}  net {r.best*100:+.2f}c")
