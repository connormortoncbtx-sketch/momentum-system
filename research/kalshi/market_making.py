"""
research/kalshi/market_making.py -- can a small account earn the spread as a passive market
maker on Kalshi daily-high temperature markets? (Oct 16, 2026)

Data: Kalshi public trade history (/historical/trades before the cutoff, /markets/trades after),
for a stratified random sample of settled markets in research_data/kalshi/markets.parquet.
Every trade has a taker side; the resting (maker) side is the opposite.

Maker P&L per contract held to settlement (R = 1 if the market settled YES):
  taker bought YES at p  -> maker sold YES at p   -> p - R
  taker bought NO        -> maker bought YES at p -> R - p
Fees: series fee_type from the API. 'quadratic' = taker fee 0.07 x p(1-p) per contract, makers
free; any '*maker*' fee type charges makers 0.0175 x p(1-p) (Kalshi schedule) -- applied if present.

Tests (dev = trades before 2025-10-01, holdout after; filters chosen on dev):
  1. Aggregate maker edge (cents per contract) by price bucket, hours before close, series.
  2. A passive book: we capture a share s of taker volume at the touch, only in filtered
     buckets, with a per-market inventory cap; P&L settles at expiry; capital = collateral
     (YES at p costs p, NO costs 1-p). Daily P&L -> return on capital, Sharpe, capacity.
Caveats: back-of-queue fills are more adversely selected than the average maker fill; we
assume our share s of each taker trade, which flatters us if better-informed flow is what
reaches the back of the queue. Report s = 2%, 5%, 10% to show sensitivity.
"""
import concurrent.futures as cf
import json, os, sys, threading, time, urllib.parse, urllib.request
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = f"{RD}/kalshi"; os.makedirs(f"{RD}/results", exist_ok=True)
API = "https://api.elections.kalshi.com/trade-api/v2"
N_SAMPLE = int(os.environ.get("N_MARKETS") or 6000)
SPLIT = pd.Timestamp("2025-10-01", tz="UTC")


class Lim:
    def __init__(self, per_sec):
        self.gap, self.lock, self.t = 1 / per_sec, threading.Lock(), 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.t:
                time.sleep(self.t - now)
            self.t = max(now, self.t) + self.gap


LIM = Lim(12)


def get(path, **q):
    url = f"{API}{path}?{urllib.parse.urlencode(q)}"
    for i in range(5):
        LIM.wait()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}
            time.sleep(2 * (i + 1))
        except Exception:
            time.sleep(2 * (i + 1))
    return {}


def trades_for(ticker, hist):
    rows, cursor = [], None
    for path in (["/historical/trades", "/markets/trades"] if hist else ["/markets/trades", "/historical/trades"]):
        while True:
            q = {"ticker": ticker, "limit": 1000}
            if cursor:
                q["cursor"] = cursor
            d = get(path, **q)
            tr = d.get("trades", [])
            rows += tr
            cursor = d.get("cursor")
            if not cursor or not tr:
                break
        if rows:
            break
    return ticker, rows


def fetch(mk):
    f = f"{OUT}/trades_sample.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    out = []
    with cf.ThreadPoolExecutor(8) as ex:
        for n, (t, rows) in enumerate(ex.map(lambda r: trades_for(r.ticker, bool(r.settled_hist)), mk.itertuples())):
            out += rows
            if n % 500 == 0:
                print(f"  {n:,}/{len(mk):,} markets, {len(out):,} trades", flush=True)
    d = pd.DataFrame(out)
    d["p"] = pd.to_numeric(d.yes_price_dollars, errors="coerce")
    d["c"] = pd.to_numeric(d.count_fp, errors="coerce")
    d["t"] = pd.to_datetime(d.created_time, utc=True, format="ISO8601")
    d = d[["ticker", "t", "p", "c", "taker_side", "is_block_trade"]]
    d.to_parquet(f, index=False)
    return d


def main():
    mk = pd.read_parquet(f"{OUT}/markets.parquet")
    mk = mk[mk.result.isin(["yes", "no"])].copy()
    mk["close_time"] = pd.to_datetime(mk.close_time, utc=True)
    mk = mk[mk.close_time >= "2024-06-01"]
    mk["month"] = mk.close_time.dt.strftime("%Y-%m")
    rng = np.random.default_rng(11)
    per = max(1, N_SAMPLE // mk.groupby(["series", "month"]).ngroups)
    smp = mk.assign(_r=rng.random(len(mk))).sort_values("_r").groupby(["series", "month"]).head(per).drop(columns="_r")
    print(f"markets sampled {len(smp):,} of {len(mk):,}", flush=True)
    fees = {s: (get(f"/series/{s}").get("series") or {}).get("fee_type", "quadratic") for s in smp.series.unique()}
    print("fee types:", fees, flush=True)
    tr = fetch(smp)
    tr = tr.merge(smp[["ticker", "series", "result", "close_time"]], on="ticker")
    tr = tr[(tr.p > 0) & (tr.p < 1) & tr.c.notna() & ~tr.is_block_trade.fillna(False)]
    tr["R"] = (tr.result == "yes").astype(float)
    tr["maker_pnl"] = np.where(tr.taker_side == "yes", tr.p - tr.R, tr.R - tr.p)       # per contract, to settlement
    maker_fee = tr.series.map(lambda s: 0.0175 if "maker" in str(fees.get(s, "")) else 0.0)
    tr["maker_net"] = tr.maker_pnl - maker_fee * tr.p * (1 - tr.p)
    tr["taker_fee"] = 0.07 * tr.p * (1 - tr.p)
    tr["hrs_left"] = (tr.close_time - tr.t).dt.total_seconds() / 3600
    tr["period"] = np.where(tr.t < SPLIT, "dev", "holdout")
    tr["pb"] = pd.cut(tr.p, [0, 0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 1], labels=["1-5c", "5-15", "15-35", "35-65", "65-85", "85-95", "95-99"])
    tr["hb"] = pd.cut(tr.hrs_left, [-1e9, 2, 6, 12, 24, 48, 1e9], labels=["<2h", "2-6h", "6-12h", "12-24h", "24-48h", ">48h"])
    pd.set_option("display.width", 220)
    w = lambda g: pd.Series({"contracts": g.c.sum(), "maker_c_per": (g.maker_net * g.c).sum() / g.c.sum() * 100,
                             "taker_fee_c": (g.taker_fee * g.c).sum() / g.c.sum() * 100})
    print(f"\ntrades {len(tr):,}, contracts {tr.c.sum():,.0f}; dev {int((tr.period == 'dev').sum()):,} / holdout {int((tr.period == 'holdout').sum()):,}")
    print("\n1. MAKER EDGE, cents per contract held to settlement (net of maker fees)")
    for by in ("pb", "hb", "series"):
        print(tr.groupby([by, "period"], observed=True).apply(w).unstack().round(2).to_string(), "\n")

    # 2. passive book: choose buckets on dev (positive maker edge with >= 50k contracts), then simulate
    g = tr[tr.period == "dev"].groupby(["pb", "hb"], observed=True).apply(w)
    keep = g[(g.maker_c_per > 0.25) & (g.contracts >= 50_000)].index.tolist()
    print("2. buckets kept on dev (maker edge > 0.25c, >= 50k contracts):", keep)
    sel = tr.set_index(["pb", "hb"]).index.isin(keep)
    X = tr[sel].copy().sort_values("t")

    def run_book(X, label):
        rows = []
        res_m = X.drop_duplicates("ticker").set_index("ticker")
        for s in (0.02, 0.05, 0.10):
            for cap in (100, 500):
                pos, cash = {}, {}
                pnl_by_day, coll_by_day = {}, {}
                for tk, t, p, c, side in zip(X.ticker, X.t, X.p, X.c, X.taker_side):
                    q = s * c
                    cur = pos.get(tk, 0.0)
                    d = -q if side == "yes" else q                     # maker sells YES when taker buys YES
                    if abs(cur + d) > cap:
                        continue
                    pos[tk] = cur + d
                    cash[tk] = cash.get(tk, 0.0) - d * p              # buying YES costs p per contract
                    day = t.normalize()
                    coll_by_day[day] = coll_by_day.get(day, 0.0) + q * (p if d > 0 else 1 - p)
                for tk, ps in pos.items():                            # settle on the market's close day
                    day = res_m.at[tk, "close_time"].normalize()
                    pnl_by_day[day] = pnl_by_day.get(day, 0.0) + cash[tk] + ps * res_m.at[tk, "R"]
                pnl = pd.Series(pnl_by_day).sort_index()
                capital = pd.Series(coll_by_day).sort_index().rolling(3, min_periods=1).sum().quantile(0.95)
                for per, m in (("dev", pnl.index < SPLIT), ("holdout", pnl.index >= SPLIT)):
                    if m.sum() < 30:
                        continue
                    x = pnl[m].reindex(pd.date_range(pnl[m].index.min(), pnl[m].index.max(), freq="D"), fill_value=0)
                    rows.append(dict(share=s, cap=cap, period=per, days=len(x), pnl_usd=x.sum(), per_month=x.sum() / len(x) * 30,
                                     capital_usd=capital, monthly_ret_pct=x.sum() / len(x) * 30 / capital * 100,
                                     sharpe_daily_ann=x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else np.nan,
                                     worst_day=x.min(), win_days=(x > 0).mean() * 100))
        for r in rows:
            r["book"] = label
        return rows

    rows = run_book(X, "real")
    # placebo: shuffle the taker side within each market (kills any maker edge, keeps volumes,
    # prices and timing) -> the Sharpe a dev-chosen book gets from noise alone
    P_ = X.copy()
    P_["taker_side"] = P_.groupby("ticker").taker_side.transform(lambda v: np.random.default_rng(5).permutation(v.values))
    rows += run_book(P_, "placebo")
    res = pd.DataFrame(rows)
    print("\n   PASSIVE BOOK (sample of markets -- scale P&L by the sampling fraction for the full market)")
    print(res.round(2).to_string(index=False))
    frac = len(smp) / len(mk)
    print(f"\n   sampling fraction {frac:.3f}: full-market P&L is roughly 1/{frac:.3f} x the sample's")
    tr.drop(columns=["pb", "hb"]).to_parquet(f"{RD}/results/kalshi_mm_trades.parquet", index=False)
    best = res[(res.period == "dev") & (res.book == "real")].sort_values("sharpe_daily_ann", ascending=False).head(1)
    if len(best):
        b = best.iloc[0]; h = res[(res.period == "holdout") & (res.share == b.share) & (res.cap == b.cap) & (res.book == "real")]
        print(f"::notice::Kalshi MM: dev-best share {b.share} cap {b.cap}: dev Sharpe {b.sharpe_daily_ann:.2f}, "
              f"holdout Sharpe {h.sharpe_daily_ann.iloc[0] if len(h) else float('nan'):.2f}, holdout monthly "
              f"{h.monthly_ret_pct.iloc[0] if len(h) else float('nan'):.1f}% on ${b.capital_usd:,.0f} (sample, frac {frac:.3f})")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
