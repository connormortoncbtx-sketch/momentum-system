"""
research/kalshi/drift.py -- do the slow, quiet-hours price drifts in daily-high markets predict where they settle?
(Oct 28, 2026; Connor: "these slow gradual movements, is there anything we can glean from these?")

For every market in the public trade sample (daily highs, ~3M trades), at anchor times A hours before close
(A = 30, 24, 18, 12, 9, 6), take the YES price then (last trade at or before, no older than 2h) and its DRIFT over the
previous L hours (L = 3, 6; price at A minus price at A+L, same staleness rule). Outcomes:
  resid   settlement (1/0) minus the anchor price, cents -- does the market under-react (drift continues: resid has the
          same sign as drift) or over-react (snaps back)?
  next3   price 3 hours later minus the anchor price, cents (the same question on a shorter, less noisy horizon)
Only anchors priced 10-90c. Drift is signed toward YES; results are also shown from the FAVOURITE's side (the side the
bot buys), so "drift +" there means the favourite has been getting more expensive.
Reported: by drift quintile (cut on DEV), the slope of resid on drift, all with 2 s.e. clustered by city-day, DEV
(< 2025-10-01) and HOLDOUT separately -- a rule is only worth having if both agree. Public data only.
Output: results/kalshi_drift.csv.
"""
import os
import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
SPLIT = pd.Timestamp("2025-10-01", tz="UTC")
ANCHORS, LOOKS, STALE = (30, 24, 18, 12, 9, 6), (3, 6), 2 * 3600
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)


def load():
    t = pd.read_parquet(os.environ.get("TRADES", f"{RD}/kalshi/trades_sample.parquet"), columns=["ticker", "t", "p", "c", "is_block_trade"])
    m = pd.read_parquet(os.environ.get("MARKETS", f"{RD}/kalshi/markets.parquet"),
                        columns=["ticker", "event_ticker", "series", "result", "close_time"])
    t = t.merge(m, on="ticker")
    t = t[t.result.isin(["yes", "no"]) & (t.p > 0) & (t.p < 1) & ~t.is_block_trade.fillna(False).astype(bool)].copy()
    t["ts"] = (pd.to_datetime(t.t, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    t["cl"] = (pd.to_datetime(t.close_time, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    return t.sort_values(["ticker", "ts"])


def snapshots(t):
    rows = []
    for tk, g in t.groupby("ticker", sort=False):
        ts, ps = g.ts.to_numpy(), g.p.to_numpy()
        cl, R, ev = g.cl.iloc[0], 1.0 if g.result.iloc[0] == "yes" else 0.0, g.event_ticker.iloc[0]

        def px(at):
            j = np.searchsorted(ts, at, side="right") - 1
            return ps[j] if j >= 0 and at - ts[j] <= STALE else np.nan
        for A in ANCHORS:
            a = cl - A * 3600
            p0 = px(a)
            if not 0.10 <= p0 <= 0.90:
                continue
            p3 = px(a + 3 * 3600) if A > 3 else np.nan
            for L in LOOKS:
                pl = px(a - L * 3600)
                if np.isnan(pl):
                    continue
                rows.append((tk, ev, A, L, a, p0, (p0 - pl) * 100, (R - p0) * 100, (p3 - p0) * 100))
    S = pd.DataFrame(rows, columns=["ticker", "event", "A", "L", "a", "p0", "drift", "resid", "next3"])
    S["when"] = pd.to_datetime(S.a, unit="s", utc=True)
    # favourite's view: flip the signs where the favourite is NO
    fav = np.where(S.p0 >= 0.5, 1.0, -1.0)
    S["fdrift"], S["fresid"], S["fnext3"] = S.drift * fav, S.resid * fav, S.next3 * fav
    return S


def mse(x, col):
    """mean and 2 s.e. clustered by city-day (equal weight per snapshot)."""
    x = x[x[col].notna()]
    if len(x) < 30:
        return np.nan, np.nan, len(x)
    g = x.groupby("event")[col].agg(["sum", "count"])
    N = g["count"].sum(); m = g["sum"].sum() / N; G = len(g)
    se = np.sqrt(((g["sum"] - m * g["count"]) ** 2).sum() / N ** 2 * G / (G - 1))
    return m, 2 * se, len(x)


def slope(x, ycol, xcol):
    """OLS slope of y on x with 2 s.e. clustered by city-day."""
    x = x[x[ycol].notna() & x[xcol].notna()]
    if len(x) < 50:
        return np.nan, np.nan
    X = x[xcol] - x[xcol].mean(); y = x[ycol] - x[ycol].mean()
    b = (X * y).sum() / (X * X).sum()
    u = X * (y - b * X)
    g = u.groupby(x.event).sum()
    se = np.sqrt((g ** 2).sum()) / (X * X).sum()
    return b, 2 * se


def main():
    t = load()
    print(f"{len(t):,} trades, {t.ticker.nunique():,} markets", flush=True)
    S = snapshots(t)
    S["part"] = np.where(S.when < SPLIT, "dev", "hold")
    print(f"{len(S):,} snapshots ({(S.part == 'dev').sum():,} dev / {(S.part == 'hold').sum():,} holdout), "
          f"{S.event.nunique():,} city-days", flush=True)
    rows = []
    for (A, L), g in S.groupby(["A", "L"]):
        d = g[g.part == "dev"]
        if len(d) < 200:
            continue
        cuts = d.fdrift.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
        for part, x in g.groupby("part"):
            q = np.searchsorted(cuts, x.fdrift.to_numpy(), side="right")
            for qi in range(5):
                xx = x[q == qi]
                m, se, n = mse(xx, "fresid"); m3, se3, _ = mse(xx, "fnext3")
                rows.append(dict(A=A, L=L, part=part, group=f"Q{qi + 1}", n=n, drift=xx.fdrift.mean(),
                                 resid_c=m, resid_2se=se, next3_c=m3, next3_2se=se3))
            b, bse = slope(x, "fresid", "fdrift"); b3, b3se = slope(x, "fnext3", "fdrift")
            rows.append(dict(A=A, L=L, part=part, group="slope", n=len(x), drift=np.nan, resid_c=b, resid_2se=bse,
                             next3_c=b3, next3_2se=b3se))
    R = pd.DataFrame(rows)
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/kalshi_drift.csv", index=False)
    print("\nFAVOURITE'S VIEW. drift = how much the favourite got more (+) or less (-) expensive over the last L hours;"
          "\nresid = settlement minus price (c/contract, + = favourite under-priced); next3 = price change over the next 3h."
          "\nslope rows: cents of resid / next3 per cent of drift (+ = drift continues, - = snaps back).\n")
    print(R.round(2).to_string(index=False))
    sl = R[R.group == "slope"].pivot_table(index=["A", "L"], columns="part", values=["resid_c", "resid_2se"])
    agree = []
    for (A, L), r in sl.iterrows():
        bd, bh = r[("resid_c", "dev")], r[("resid_c", "hold")]
        sd, sh = r[("resid_2se", "dev")], r[("resid_2se", "hold")]
        if np.sign(bd) == np.sign(bh) and abs(bd) > sd and abs(bh) > sh:
            agree.append(f"{A}h/{L}h {bd:+.2f}/{bh:+.2f}")
    print(f"::notice::drift study: {len(S):,} snapshots; slope of settlement residual on drift (dev/hold) significant "
          f"and same sign in both: {', '.join(agree) if agree else 'none'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
