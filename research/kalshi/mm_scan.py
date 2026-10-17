"""
research/kalshi/mm_scan.py -- where on Kalshi do passive makers earn, and is it the NEW series? (Oct 17, 2026)

Same measurement as research/kalshi/market_making.py (maker side of every public trade held
to settlement, net of maker fees where the series charges them), applied across ALL series:
  1. list every series (category, fee_type) and its settled markets (live + historical)
  2. per series with enough volume, sample markets from its FIRST 90 days and its LAST 90 days
  3. fetch their trades; maker edge in cents/contract, and a t-stat clustered by market
     (each market's contract-weighted edge is one observation)
Outputs results/kalshi_mm_scan.csv (one row per series) and a printed summary by category
and by series age. NEW_ONLY=1 restricts to series whose first market closed in the last 120
days -- the weekly new-series watch.
Caveats: same as the weather study -- this is the AVERAGE maker's edge; back-of-queue fills
are worse; collateral is tied up by resting orders.
"""
import concurrent.futures as cf
import json, os, threading, time, urllib.error, urllib.parse, urllib.request
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = f"{RD}/kalshi/scan"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{RD}/results", exist_ok=True)
API = "https://api.elections.kalshi.com/trade-api/v2"
PER_WINDOW = int(os.environ.get("PER_WINDOW") or 25)      # markets sampled per (series, early/late)
MIN_VOL = float(os.environ.get("MIN_VOL") or 20_000)       # contracts traded across settled markets
NEW_ONLY = os.environ.get("NEW_ONLY") == "1"
BUDGET_S = float(os.environ.get("BUDGET_MIN") or 300) * 60
T0 = time.time()


class Lim:
    def __init__(self, per_sec):
        self.gap, self.lock, self.t = 1 / per_sec, threading.Lock(), 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.t:
                time.sleep(self.t - now)
            self.t = max(now, self.t) + self.gap


LIM = Lim(15)


def get(path, **q):
    url = f"{API}{path}?{urllib.parse.urlencode(q)}"
    for i in range(5):
        LIM.wait()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return {}
            time.sleep(2 * (i + 1))
        except Exception:
            time.sleep(2 * (i + 1))
    return {}


def paged(path, key, max_pages=50, **q):
    rows, cursor = [], None
    for _ in range(max_pages):
        if cursor:
            q["cursor"] = cursor
        d = get(path, limit=1000 if "trades" in path else 1000, **q)
        rows += d.get(key, []) or []
        cursor = d.get("cursor")
        if not cursor or not d.get(key):
            break
    return rows


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def series_list():
    p = f"{OUT}/series.parquet"
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < 86400 * 3:
        return pd.read_parquet(p)
    s = pd.DataFrame(paged("/series", "series", max_pages=100))
    keep = [c for c in ("ticker", "title", "category", "frequency", "fee_type", "fee_multiplier", "tags") if c in s.columns]
    s = s[keep].copy()
    for c in s.columns:
        s[c] = s[c].astype(str)
    s.to_parquet(p, index=False)
    return s


def markets_for(series):
    rows = paged("/markets", "markets", max_pages=20, series_ticker=series, status="settled") + \
           paged("/historical/markets", "markets", max_pages=20, series_ticker=series)
    out = []
    for m in rows:
        if m.get("result") not in ("yes", "no"):
            continue
        out.append(dict(ticker=m["ticker"], series=series, close_time=m.get("close_time"), result=m["result"],
                        volume=f(m.get("volume_fp", m.get("volume")))))
    d = pd.DataFrame(out).drop_duplicates("ticker") if out else pd.DataFrame()
    if len(d):
        d["close_time"] = pd.to_datetime(d.close_time, utc=True, errors="coerce")
    return d


def trades_for(ticker):
    rows = []
    for path in ("/markets/trades", "/historical/trades"):
        rows = paged(path, "trades", max_pages=10, ticker=ticker)
        if rows:
            break
    return [dict(ticker=ticker, t=r.get("created_time"), p=f(r.get("yes_price_dollars")), c=f(r.get("count_fp", r.get("count"))),
                 taker_side=r.get("taker_side"), block=bool(r.get("is_block_trade"))) for r in rows]


def scan_series(s, fee_type, rng):
    cache = f"{OUT}/s_{s}.parquet"
    if os.path.exists(cache) and not NEW_ONLY:
        return pd.read_parquet(cache)
    mk = markets_for(s)
    if not len(mk) or mk.volume.sum(skipna=True) < MIN_VOL:
        pd.DataFrame({"ticker": []}).to_parquet(cache, index=False)
        return None
    first, last = mk.close_time.min(), mk.close_time.max()
    mk["window"] = np.where(mk.close_time <= first + pd.Timedelta(days=90), "early",
                            np.where(mk.close_time >= last - pd.Timedelta(days=90), "late", "mid"))
    mk = mk[(mk.window != "mid") & (mk.volume > 0)]
    smp = mk.groupby("window", group_keys=False).apply(
        lambda g: g.sample(min(len(g), PER_WINDOW), weights=g.volume.fillna(0) + 1, random_state=int(rng.integers(1e9))))
    tr = []
    for tk in smp.ticker:
        tr += trades_for(tk)
    if not tr:
        return None
    t = pd.DataFrame(tr).merge(smp[["ticker", "series", "result", "close_time", "window"]], on="ticker")
    t = t[(t.p > 0) & (t.p < 1) & t.c.notna() & ~t.block]
    t["R"] = (t.result == "yes").astype(float)
    mfee = 0.0175 if "maker" in str(fee_type) else 0.0
    t["edge"] = np.where(t.taker_side == "yes", t.p - t.R, t.R - t.p) - mfee * t.p * (1 - t.p)
    t["first"], t["last"] = first, last
    t.to_parquet(cache, index=False)
    return t


def summarise(t):
    m = t.assign(ec=t.edge * t.c).groupby(["series", "window", "ticker"]).agg(ec=("ec", "sum"), c=("c", "sum")).reset_index()
    m["e"] = m.ec / m.c
    rows = []
    for (s, w), g in m.groupby(["series", "window"]):
        e = g.ec.sum() / g.c.sum() * 100
        se = g.e.std(ddof=1) / np.sqrt(len(g)) * 100 if len(g) > 2 else np.nan
        rows.append(dict(series=s, window=w, markets=len(g), contracts=g.c.sum(), edge_c=e, t=e / se if se else np.nan))
    return pd.DataFrame(rows)


def main():
    S = series_list()
    print(f"series listed: {len(S):,}; categories: {S.category.value_counts().head(15).to_dict()}", flush=True)
    rng = np.random.default_rng(17)
    fees = dict(zip(S.ticker, S.get("fee_type", pd.Series("", index=S.index))))
    todo = list(S.ticker)
    rng.shuffle(todo)
    parts, n_done = [], 0

    def job(s):
        if time.time() - T0 > BUDGET_S:
            return None
        try:
            return scan_series(s, fees.get(s), np.random.default_rng(abs(hash(s)) % 2**32))
        except Exception as e:
            print(f"  {s}: {e}", flush=True)
            return None

    with cf.ThreadPoolExecutor(6) as ex:
        for t in ex.map(job, todo):
            n_done += 1
            if t is not None and len(t) and "edge" in t:
                if NEW_ONLY and t["first"].iloc[0] < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=120):
                    continue
                parts.append(t)
            if n_done % 200 == 0:
                print(f"  {n_done:,}/{len(todo):,} series, {len(parts)} with volume, {(time.time() - T0) / 60:.0f} min", flush=True)
    print(f"scanned {n_done:,} series ({'budget hit' if time.time() - T0 > BUDGET_S else 'complete'}); {len(parts)} with data")
    if not parts:
        return
    T = pd.concat(parts, ignore_index=True)
    R = summarise(T).merge(S.rename(columns={"ticker": "series"}), on="series", how="left")
    age = T.groupby("series").agg(first=("first", "first"), last=("last", "last")).reset_index()
    R = R.merge(age, on="series")
    R.to_csv(f"{RD}/results/kalshi_mm_scan{'_new' if NEW_ONLY else ''}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)

    def wavg(g):
        return pd.Series({"series": g.series.nunique(), "contracts_M": g.contracts.sum() / 1e6,
                          "edge_c": (g.edge_c * g.contracts).sum() / g.contracts.sum(),
                          "share_series_pos": (g.edge_c > 0).mean() * 100})
    print("\nBY CATEGORY x WINDOW (early = first 90 days of the series, late = last 90 days)")
    print(R.groupby(["category", "window"]).apply(wavg).round(2).to_string())
    late = R[R.window == "late"]
    print("\nTOP SERIES, LATE WINDOW (edge > 0.5c, t > 2, >= 100k contracts in sample)")
    print(late[(late.edge_c > 0.5) & (late.t > 2) & (late.contracts >= 1e5)].sort_values("edge_c", ascending=False)
          [["series", "category", "title", "fee_type", "markets", "contracts", "edge_c", "t", "first"]].head(40).to_string(index=False))
    both = R.pivot_table(index="series", columns="window", values="edge_c")
    if {"early", "late"} <= set(both.columns):
        b = both.dropna()
        print(f"\nEARLY vs LATE, same series (n={len(b)}): early mean {b.early.mean():.2f}c, late {b.late.mean():.2f}c, "
              f"late < early in {(b.late < b.early).mean() * 100:.0f}% of series")
    T["since_launch_d"] = (T.close_time - T["first"]).dt.days
    T["age_bin"] = pd.cut(T.since_launch_d, [-1, 30, 90, 180, 365, 10_000], labels=["0-30d", "31-90d", "91-180d", "181-365d", ">1y"])
    print("\nEDGE BY SERIES AGE AT THE TRADE (all series pooled, c/contract)")
    print(T.groupby("age_bin", observed=True).apply(lambda g: pd.Series({"contracts_M": g.c.sum() / 1e6,
          "edge_c": (g.edge * g.c).sum() / g.c.sum() * 100})).round(2).to_string())
    top = late[(late.edge_c > 0.5) & (late.t > 2) & (late.contracts >= 1e5)].sort_values("edge_c", ascending=False).head(5)
    print("::notice::Kalshi MM scan: " + "; ".join(f"{r.series} {r.edge_c:.1f}c t{r.t:.1f}" for r in top.itertuples()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
