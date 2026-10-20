"""
research/pm_copytrade.py -- does copying top prediction-market traders work? (Polymarket proxy, Oct 20, 2026)

Kalshi's data terms forbid harvesting its website (profiles), so this uses Polymarket's public data API (every trade
is on-chain). Question: do traders who looked skilled in one period keep an edge we can capture by copying them
LATE (5 / 15 / 60 minutes after their trade, at the then-current price + 1c)?

  sampling   per wallet, the latest 500 trades of each 30-day slice
  pool       top 400 wallets by all-time VOLUME (volume, not profit, so the pool isn't picked on outcomes)
  formation  Oct 20 2025 - Apr 20 2026: each wallet's BUY trades in resolved markets -> edge per share
             (payout - price); score = mean edge, >= 40 trades
  test       Apr 20 - Oct 6 2026 (resolved markets only):
               PERSIST   the top-K wallets' own BUY edge in the test period (would their trades have paid?)
               COPY      buying the same outcome token LAG minutes later at the price-history price + 1c
               CONSENSUS a token bought by >= 3 top-25 wallets within 24h: copy LAG after the 3rd buy
             vs the bottom half of the pool (control), and vs all pool BUY trades
Prices 0.03-0.97 only. Polymarket international has ~0 taker fees; +1c stands in for the spread. Polymarket US and
Kalshi charge fees, so a copy edge must clear ~1-2c more there.
Raw pulls cached under research_data/polymarket/ (resume-safe).
"""
import concurrent.futures as cf, json, os, threading, time, traceback, urllib.request
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = f"{RD}/polymarket" + ("_smoke" if os.environ.get("LIMIT") else ""); os.makedirs(OUT, exist_ok=True); os.makedirs(f"{RD}/results", exist_ok=True)
D, G, C = "https://data-api.polymarket.com", "https://gamma-api.polymarket.com", "https://clob.polymarket.com"
F0, F1, T1 = 1760918400, 1776643200, 1791244800          # 2025-10-20, 2026-04-20, 2026-10-06 (UTC)
POOL, LAGS = int(os.environ.get("LIMIT") or os.environ.get("POOL") or 400), (5, 15, 60)   # LIMIT = smoke test
UA = {"User-Agent": "Mozilla/5.0 research", "Accept": "application/json"}
_lock = threading.Lock(); _last = [0.0]


def get(url, tries=5):
    for i in range(tries):
        with _lock:
            w = 0.06 - (time.time() - _last[0])
            if w > 0: time.sleep(w)
            _last[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1) + (10 if "429" in str(e) else 0))


def pool():
    p = f"{OUT}/pool.json"
    if os.path.exists(p):
        return json.load(open(p))
    w = []
    for off in range(0, POOL, 50):
        w += get(f"{D}/v1/leaderboard?timePeriod=ALL&orderBy=VOL&limit=50&offset={off}")
    w = [x["proxyWallet"] for x in w][:POOL]
    json.dump(w, open(p, "w"))
    return w


def wallet_trades(addr):
    """TRADE activity F0..T1: the latest 500 trades of each ~30-day slice (offset paging stops at ~3000, and very
    active wallets have far more), so both periods are evenly covered."""
    p = f"{OUT}/w_{addr}.parquet"
    if os.path.exists(p):
        return pd.read_parquet(p)
    rows = []
    for s0 in range(F0, T1, 30 * 86400):
        e0 = min(s0 + 30 * 86400, T1) - 1
        try:
            d = get(f"{D}/activity?user={addr}&type=TRADE&limit=500&start={s0}&end={e0}&sortBy=TIMESTAMP&sortDirection=DESC")
        except Exception as e:
            print(f"::warning::{addr[:10]} activity: {str(e)[:120]}"); continue
        rows += [{k: x.get(k) for k in ("timestamp", "conditionId", "asset", "side", "price", "size", "usdcSize", "outcome", "title")}
                 for x in d or []]
    df = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["timestamp", "side"])
    df["wallet"] = addr
    df.to_parquet(p, index=False)
    return df


def resolutions(cids):
    """conditionId -> {asset: payout} for closed markets."""
    p = f"{OUT}/resolutions.json"
    res = json.load(open(p)) if os.path.exists(p) else {}
    need = [c for c in cids if c and c not in res]
    for i in range(0, len(need), 40):
        q = "&".join(f"condition_ids={c}" for c in need[i:i + 40])
        try:
            ms = get(f"{G}/markets?{q}&limit=100&closed=true")
        except Exception as e:
            print(f"::warning::gamma: {str(e)[:120]}"); continue
        for m in ms:
            try:
                toks = json.loads(m.get("clobTokenIds") or "[]"); px = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
            except Exception:
                continue
            if m.get("closed") and len(toks) == len(px) and px and max(px) > 0.99:
                res[m["conditionId"]] = dict(zip(toks, px))
        for c in need[i:i + 40]:
            res.setdefault(c, None)
        if i % 2000 == 0:
            json.dump(res, open(p, "w"))
    json.dump(res, open(p, "w"))
    return res


def later_prices(asset, ts):
    """Price-history price at ts + each lag (first print at/after), or None."""
    try:
        h = get(f"{C}/prices-history?market={asset}&startTs={ts}&endTs={ts + 90 * 60}&fidelity=1").get("history", [])
    except Exception:
        return {}
    out = {}
    for L in LAGS:
        tgt = ts + L * 60
        v = [x["p"] for x in h if x["t"] >= tgt]
        if v:
            out[L] = float(v[0])
    return out


def edge_stats(e, key=None):
    e = pd.Series(e).dropna()
    return dict(n=len(e), edge_c=e.mean() * 100, t=e.mean() / e.std() * np.sqrt(len(e)) if len(e) > 2 and e.std() > 0 else np.nan,
                win_pct=(e > 0).mean() * 100)


def main():
    W = pool()
    print(f"pool {len(W)} wallets", flush=True)
    with cf.ThreadPoolExecutor(6) as ex:
        T = pd.concat(list(ex.map(wallet_trades, W)), ignore_index=True)
    T = T[T.side.isin(["BUY", "SELL"])].copy()
    T["timestamp"] = T.timestamp.astype(int); T["price"] = T.price.astype(float)
    print(f"trades {len(T):,} from {T.wallet.nunique()} wallets", flush=True)
    res = resolutions(sorted(set(T.conditionId.dropna())))
    T["payout"] = [((res.get(c) or {}).get(a)) for c, a in zip(T.conditionId, T.asset)]
    B = T[(T.side == "BUY") & T.payout.notna() & T.price.between(0.03, 0.97)].copy()
    B["edge"] = B.payout - B.price
    print(f"resolved BUY trades {len(B):,}", flush=True)
    form, test = B[B.timestamp < F1], B[B.timestamp >= F1]
    sc = form.groupby("wallet").agg(n=("edge", "size"), score=("edge", "mean"))
    sc = sc[sc.n >= 40].sort_values("score", ascending=False)
    sc.to_csv(f"{RD}/results/pm_copy_scores.csv")
    rows = []
    groups = {"top10": list(sc.index[:10]), "top25": list(sc.index[:25]), "top50": list(sc.index[:50]),
              "bottom half": list(sc.index[len(sc) // 2:]), "all pool": list(sc.index)}
    for gname, ws in groups.items():
        x = test[test.wallet.isin(ws)]
        rows.append(dict(group=gname, wallets=len(ws), measure="own trades (persistence)", **edge_stats(x.edge)))
        # also $-weighted
        rows.append(dict(group=gname, wallets=len(ws), measure="own trades, $-weighted",
                         n=len(x), edge_c=(x.edge * x.usdcSize.astype(float)).sum() / max(x.usdcSize.astype(float).sum(), 1e-9) * 100,
                         t=np.nan, win_pct=np.nan))
    # COPY with delay: sample up to 150 test trades per wallet for the top-50 and the bottom half (price-history calls)
    rng = np.random.default_rng(7)
    samp = []
    for gname in ("top50", "bottom half"):
        x = test[test.wallet.isin(groups[gname])]
        x = x.groupby("wallet", group_keys=False).apply(lambda g: g.sample(min(len(g), 150), random_state=7))
        samp.append(x.assign(group=gname))
    S = pd.concat(samp, ignore_index=True)
    cache = f"{OUT}/later_prices.json"
    lp = json.load(open(cache)) if os.path.exists(cache) else {}
    keys = [f"{a}|{t}" for a, t in zip(S.asset, S.timestamp)]
    need = [k for k in set(keys) if k not in lp]
    print(f"price-history lookups: {len(need):,}", flush=True)
    def look(k):
        a, t = k.split("|"); return k, later_prices(a, int(t))
    with cf.ThreadPoolExecutor(6) as ex:
        for i, (k, v) in enumerate(ex.map(look, need)):
            lp[k] = {str(L): p for L, p in v.items()}
            if i % 2000 == 0:
                json.dump(lp, open(cache, "w"))
    json.dump(lp, open(cache, "w"))
    for L in LAGS:
        S[f"copy{L}"] = [lp.get(k, {}).get(str(L)) for k in keys]
        S[f"copy{L}"] = S.payout - (S[f"copy{L}"].astype(float) + 0.01)
    for gname in ("top10", "top25", "top50", "bottom half"):
        x = S[S.wallet.isin(groups[gname])]
        rows.append(dict(group=gname, wallets=len(groups[gname]), measure="their own price (sampled)", **edge_stats(x.edge)))
        for L in LAGS:
            rows.append(dict(group=gname, wallets=len(groups[gname]), measure=f"COPY +{L}min +1c", **edge_stats(x[f"copy{L}"])))
    # CONSENSUS: >= 3 distinct top-25 wallets buy the same token within 24h -> copy at the 3rd buy + lag
    x = test[test.wallet.isin(groups["top25"])].sort_values("timestamp")
    sig = []
    for a, g in x.groupby("asset"):
        g = g.drop_duplicates("wallet")
        if len(g) >= 3:
            t3 = g.timestamp.iloc[2]
            if t3 - g.timestamp.iloc[0] <= 86400:
                sig.append((a, int(t3), float(g.payout.iloc[0]), float(g.price.iloc[2])))
    cons = []
    for a, t3, pay, p3 in sig:
        v = later_prices(a, t3)
        cons.append(dict(asset=a, at=t3, edge_3rd=pay - p3, **{f"copy{L}": (pay - v[L] - 0.01) if L in v else np.nan for L in LAGS}))
    Cn = pd.DataFrame(cons)
    if len(Cn):
        rows.append(dict(group="top25 consensus (>=3 in 24h)", wallets=25, measure="3rd buyer's price", **edge_stats(Cn.edge_3rd)))
        for L in LAGS:
            rows.append(dict(group="top25 consensus (>=3 in 24h)", wallets=25, measure=f"COPY +{L}min +1c", **edge_stats(Cn[f"copy{L}"])))
    R = pd.DataFrame(rows)
    R.to_csv(f"{RD}/results/pm_copytrade.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.round(2).to_string(index=False))
    for r in R[R.measure.str.startswith("COPY +15") | (R.measure == "own trades (persistence)")].itertuples():
        print(f"::notice::{r.group} | {r.measure}: {r.edge_c:+.2f}c/share (t {r.t:.1f}, n {r.n})")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        for line in traceback.format_exc().strip().splitlines()[-10:]:
            print(f"::error::{line}")
        raise
