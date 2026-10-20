"""
research/pm_copytrade2.py -- copy-trading v2: a pool built from the formation period itself, analysed by size band
(Oct 20, 2026). v1 (pm_copytrade.py) used the top 400 wallets by all-time volume, which favours whales, market makers
and high-frequency accounts; a skilled low-volume specialist would not be in it.

  t-stats    clustered by wallet (a wallet's trades are not independent)
  pool       wallets that TOOK trades in a broad sample of markets that closed Oct 20 2025 - Apr 20 2026 (Gamma
             closed markets, >= $5k volume), seen in >= 3 sampled markets; nothing about Apr-Oct 2026 is used
  history    per wallet, the latest 500 trades of each 30-day slice Oct 2025 - Oct 2026 (pm_copytrade.wallet_trades)
  bands      formation BUY dollars: < $2k, $2-20k, $20-200k, $200k-2M, > $2M
  per band   wallets with >= 30 formation BUYs in resolved markets, ranked by formation t-stat of edge per share
             (payout - price); test Apr 20 - Oct 6 2026:
               PERSIST  top 10% vs bottom half: own edge per share in the test period, correlation of skill
               COPY     top 15 per band, up to 100 test BUYs each: buy LAG (5 / 15 / 60 min) later at price-history + 1c
  watchlist  wallets in the top 10% of their band in formation AND with a positive test edge on >= 30 trades
             (candidates for a recurring informed-trader monitor -- only meaningful if the band-level test holds)
"""
import concurrent.futures as cf, json, os, random, traceback
import numpy as np, pandas as pd
import pm_copytrade as P

RD = P.RD
OUT = P.OUT
G, D = P.G, P.D
BANDS = [0, 2e3, 2e4, 2e5, 2e6, np.inf]
BLAB = ["<$2k", "$2-20k", "$20-200k", "$200k-2M", ">$2M"]
NMARK = int(os.environ.get("LIMIT") or 1500)


def formation_markets():
    p = f"{OUT}/v2_markets.json"
    if os.path.exists(p):
        return json.load(open(p))
    ms = []
    edges = pd.date_range("2025-10-20", "2026-04-20", freq="15D").append(pd.DatetimeIndex(["2026-04-20"]))
    for a, b in zip(edges[:-1], edges[1:]):          # half-month windows: Gamma stops paging at offset ~2500
        for off in range(0, 2500, 500):
            try:
                d = P.get(f"{G}/markets?closed=true&end_date_min={a:%Y-%m-%d}T00:00:00Z&end_date_max={b:%Y-%m-%d}T00:00:00Z"
                          f"&volume_num_min=5000&limit=500&offset={off}")
            except Exception as e:
                print(f"::warning::gamma {a:%Y-%m-%d} page {off}: {str(e)[:100]}"); break
            if not d:
                break
            ms += [m["conditionId"] for m in d if m.get("conditionId")]
            if len(d) < 500:
                break
    ms = sorted(set(ms))
    random.Random(7).shuffle(ms)
    ms = ms[:NMARK]
    json.dump(ms, open(p, "w"))
    print(f"formation markets: sampled {len(ms)}", flush=True)
    return ms


def market_wallets(cid):
    out = []
    for off in (0, 500, 1000):
        try:
            d = P.get(f"{D}/trades?market={cid}&limit=500&offset={off}")
        except Exception:
            break
        out += [x.get("proxyWallet") for x in d or [] if x.get("proxyWallet")]
        if not d or len(d) < 500:
            break
    return cid, set(out)


def build_pool():
    p = f"{OUT}/v2_pool.json"
    if os.path.exists(p):
        return json.load(open(p))
    ms = formation_markets()
    cnt = {}
    with cf.ThreadPoolExecutor(6) as ex:
        for cid, ws in ex.map(market_wallets, ms):
            for w in ws:
                cnt[w] = cnt.get(w, 0) + 1
    pool = sorted(w for w, c in cnt.items() if c >= 3)
    cap = int(os.environ.get("POOL2", 2500))
    if len(pool) > cap:
        pool = random.Random(7).sample(pool, cap)
    json.dump(pool, open(p, "w"))
    print(f"pool: {len(cnt):,} wallets seen, {len(pool):,} kept (>= 3 sampled markets, cap {cap})", flush=True)
    return pool


def main():
    pool = build_pool()
    with cf.ThreadPoolExecutor(6) as ex:
        T = pd.concat(list(ex.map(P.wallet_trades, pool)), ignore_index=True)
    T = T[T.side.isin(["BUY", "SELL"])].copy()
    T["timestamp"] = T.timestamp.astype(int); T["price"] = T.price.astype(float); T["usdcSize"] = T.usdcSize.astype(float)
    res = P.resolutions(sorted(set(T.conditionId.dropna())))
    T["payout"] = [((res.get(c) or {}).get(a)) for c, a in zip(T.conditionId, T.asset)]
    B = T[(T.side == "BUY") & T.payout.notna() & T.price.between(0.03, 0.97)].copy()
    B["edge"] = B.payout.astype(float) - B.price
    f, t = B[B.timestamp < P.F1], B[B.timestamp >= P.F1]
    print(f"trades {len(T):,}; resolved BUYs formation {len(f):,} / test {len(t):,}", flush=True)
    g = f.groupby("wallet").agg(n=("edge", "size"), mean=("edge", "mean"), sd=("edge", "std"), usd=("usdcSize", "sum"))
    g = g[g.n >= 30]
    g["tstat"] = g["mean"] / g.sd.replace(0, np.nan) * np.sqrt(g.n)
    g["band"] = pd.cut(g.usd, BANDS, labels=BLAB)
    te = t.groupby("wallet").agg(tn=("edge", "size"), tmean=("edge", "mean"), tusd=("usdcSize", "sum"))
    g = g.join(te)
    g.to_csv(f"{RD}/results/pm_copy2_wallets.csv")
    rows, copyw = [], {}
    for b in BLAB:
        x = g[(g.band == b)].dropna(subset=["tstat"]).sort_values("tstat", ascending=False)
        if len(x) < 20:
            rows.append(dict(band=b, wallets=len(x))); continue
        k = max(5, len(x) // 10)
        top, bot = x.index[:k], x.index[len(x) // 2:]
        tt, tb = t[t.wallet.isin(top)], t[t.wallet.isin(bot)]
        pe = tt.groupby("wallet").edge.mean()
        rows.append(dict(band=b, wallets=len(x), top_k=k, form_top_c=x["mean"][:k].mean() * 100,
                         test_top_c=tt.edge.mean() * 100,
                         test_top_t=tt.edge.mean() / tt.edge.std() * np.sqrt(len(tt)) if len(tt) > 2 else np.nan,
                         test_top_trades=len(tt), top_wallets_positive=(pe > 0).mean() * 100,
                         test_top_wallet_t=pe.mean() / pe.std() * np.sqrt(len(pe)) if len(pe) > 2 else np.nan,
                         test_bottom_c=tb.edge.mean() * 100,
                         skill_corr=x.tstat.corr(x.tmean)))
        copyw[b] = list(x.index[:15])
    R = pd.DataFrame(rows)
    # COPY test per band
    samp = []
    for b, ws in copyw.items():
        x = t[t.wallet.isin(ws)].sample(frac=1, random_state=7).groupby("wallet").head(100)
        samp.append(x.assign(band=b))
    crow = []
    if samp:
        S = pd.concat(samp, ignore_index=True)
        cache = f"{OUT}/later_prices.json"
        lp = json.load(open(cache)) if os.path.exists(cache) else {}
        keys = [f"{a}|{ts}" for a, ts in zip(S.asset, S.timestamp)]
        need = [k for k in set(keys) if k not in lp]
        print(f"price-history lookups: {len(need):,}", flush=True)

        def look(k):
            a, ts = k.split("|"); return k, P.later_prices(a, int(ts))
        with cf.ThreadPoolExecutor(6) as ex:
            for i, (k, v) in enumerate(ex.map(look, need)):
                lp[k] = {str(L): p for L, p in v.items()}
                if i % 2000 == 0:
                    json.dump(lp, open(cache, "w"))
        json.dump(lp, open(cache, "w"))
        for L in P.LAGS:
            px = pd.Series([lp.get(k, {}).get(str(L)) for k in keys], dtype=float)
            S[f"c{L}"] = S.payout.astype(float).values - (px.values + 0.01)
        for b, x in S.groupby("band"):
            r = dict(band=b, copy_trades=len(x), copy_wallets=x.wallet.nunique(), own_c=x.edge.mean() * 100)
            for L in P.LAGS:
                e = x[f"c{L}"].dropna()
                r[f"copy{L}_c"] = e.mean() * 100
                pw = x.dropna(subset=[f"c{L}"]).groupby("wallet")[f"c{L}"].mean()   # clustered by wallet
                r[f"copy{L}_t"] = pw.mean() / pw.std() * np.sqrt(len(pw)) if len(pw) > 2 else np.nan
                r[f"copy{L}_wallets_pos"] = (pw > 0).mean() * 100
            crow.append(r)
    Cp = pd.DataFrame(crow)
    W = g[(g.tn >= 30) & (g.tmean > 0)].copy()
    W["top_in_band"] = W.index.isin([w for b in BLAB for w in g[g.band == b].sort_values("tstat", ascending=False).index[:max(5, (g.band == b).sum() // 10)]])
    W = W[W.top_in_band].sort_values("tmean", ascending=False)
    W.to_csv(f"{RD}/results/pm_copy2_watchlist.csv")
    R.to_csv(f"{RD}/results/pm_copy2_bands.csv", index=False); Cp.to_csv(f"{RD}/results/pm_copy2_copy.csv", index=False)
    pd.set_option("display.width", 240)
    print("PERSISTENCE BY SIZE BAND (cents per share)\n" + R.round(2).to_string(index=False))
    print("\nCOPY TOP 15 PER BAND\n" + Cp.round(2).to_string(index=False))
    print(f"\nwatchlist candidates: {len(W)}")
    for r in R.dropna(subset=["test_top_c"]).itertuples():
        c = Cp[Cp.band == r.band]
        cc = f"; copy +15min {c.copy15_c.iloc[0]:+.1f}c (wallet-t {c.copy15_t.iloc[0]:.1f})" if len(c) else ""
        print(f"::notice::{r.band}: {r.wallets} wallets; top {r.top_k} formation {r.form_top_c:+.1f}c -> test "
              f"{r.test_top_c:+.2f}c (wallet-t {r.test_top_wallet_t:.1f}, {r.top_wallets_positive:.0f}% of wallets +) vs bottom half {r.test_bottom_c:+.2f}c; skill corr {r.skill_corr:.2f}{cc}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        for line in traceback.format_exc().strip().splitlines()[-10:]:
            print(f"::error::{line}")
        raise
