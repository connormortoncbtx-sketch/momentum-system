"""
research/kalshi/minute_window.py -- do weather-market fills around the hourly observation lose? (Oct 22, 2026)

Live hint (kmm diagnose, old long-shot-inclusive logic, 777 settled contracts): fills at :45-:59 past the hour settled
-7 to -8c per contract vs +3 to +6c at :15-:44. Hourly station readings (METAR) publish ~:51-:56, and 5-minute ASOS data
earlier, so takers in those minutes may be better informed. Pre-specified hypothesis + rule: pull quotes :45-:59.

Tested on the public trade archive (all weather series in the sample), dev < 2025-10-01 <= holdout:
  A. maker edge at the touch, per taker trade, by minute bucket -- if takers in :45-:59 are informed, the maker side
     earns less there. Split by the side the maker ends up holding (favourite >= 50c / long shot < 50c).
  B. the bot's simulated deep fills (profit_lock.fills: sweep reaching 2c behind, 5-lot, 6-48h, 25/market cap), held
     to settlement, by minute bucket; favourite-only subset = what the live bot now does.
Edge in cents per contract; t from per-city-day means (fills in one city-day are one bet).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from mm_perfill import load, SPLIT
from profit_lock import fills

pd.set_option("display.width", 220)
BUCKETS = [(0, 15, ":00-:14"), (15, 30, ":15-:29"), (30, 45, ":30-:44"), (45, 51, ":45-:50"), (51, 60, ":51-:59")]


def bucket(t):
    m = t.dt.minute
    out = pd.Series("", index=t.index)
    for lo, hi, lab in BUCKETS:
        out[(m >= lo) & (m < hi)] = lab
    return out


def stats(g):
    ev = g.ticker.str.rsplit("-", n=1).str[0]
    pe = (g.e * g.c).groupby(ev).sum() / g.c.groupby(ev).sum()
    return pd.Series(dict(city_days=len(pe), contracts=g.c.sum(), edge_c=(g.e * g.c).sum() / g.c.sum() * 100,
                          per_cityday_c=pe.mean() * 100,
                          t=pe.mean() / pe.std() * np.sqrt(len(pe)) if len(pe) > 2 and pe.std() > 0 else np.nan))


def diff_t(g, window):
    """city-day clustered test of (edge in window) - (edge outside), via per-city-day difference where both exist."""
    ev = g.ticker.str.rsplit("-", n=1).str[0]
    w = g.mb.isin(window)
    a = (g.e * g.c)[w].groupby(ev[w]).sum() / g.c[w].groupby(ev[w]).sum()
    b = (g.e * g.c)[~w].groupby(ev[~w]).sum() / g.c[~w].groupby(ev[~w]).sum()
    d = (a - b).dropna()
    return d.mean() * 100, (d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 2 else np.nan), len(d)


def show(name, x):
    x = x.assign(period=np.where(x.t < SPLIT, "dev", "holdout"), mb=bucket(x.t),
                 side=np.where(x.hp >= 0.5, "favourite", "longshot"))
    print(f"\n{name}")
    for side, xs in [("all", x), ("favourite", x[x.side == "favourite"]), ("longshot", x[x.side == "longshot"])]:
        r = xs.groupby(["period", "mb"]).apply(stats).round(2)
        print(f"\n  held side: {side}\n" + r.to_string())
    res = {}
    print("\n  RULE skip :45-:59 -- edge all vs kept, and within-city-day difference (window - rest):")
    for side, xs in [("all", x), ("favourite", x[x.side == "favourite"])]:
        for per, g in xs.groupby("period"):
            keep = g[~g.mb.isin([":45-:50", ":51-:59"])]
            d, dt_, n = diff_t(g, [":45-:50", ":51-:59"])
            res[(side, per)] = (d, dt_)
            print(f"   {side:9} {per:8} all {(g.e * g.c).sum() / g.c.sum() * 100:+.2f}c | kept {(keep.e * keep.c).sum() / keep.c.sum() * 100:+.2f}c "
                  f"({keep.c.sum() / g.c.sum() * 100:.0f}% of contracts, {(keep.e * keep.c).sum() / (g.e * g.c).sum() * 100 if (g.e * g.c).sum() else np.nan:.0f}% of P&L) "
                  f"| window - rest {d:+.2f}c t {dt_:.1f} over {n} city-days")
    return res


def main():
    t = load()
    a = t[(t.h >= 6) & (t.h <= 48) & (t.p >= 0.03) & (t.p <= 0.97)].copy()
    a["e"] = np.where(a.taker_side == "yes", a.p - a.R, a.R - a.p)          # maker's edge per contract
    a["hp"] = np.where(a.taker_side == "yes", 1 - a.p, a.p)                 # price of the side the maker holds
    ra = show("A. MAKER EDGE AT THE TOUCH (taker trades 6-48h before close), by minute of the hour", a)

    F = fills(t)
    F["c"] = F.q.abs()
    F["e"] = np.where(F.q > 0, F.R - F.px, F.px - F.R)
    F["hp"] = np.where(F.q > 0, F.px, 1 - F.px)
    rb = show("B. BOT'S SIMULATED DEEP FILLS held to settlement, by minute of the hour", F)

    out = os.path.join(os.environ.get("RESEARCH_DATA", "research_data"), "results")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame([dict(test=k0, side=k[0], period=k[1], window_minus_rest_c=v[0], t=v[1])
                  for k0, r in (("touch", ra), ("deep", rb)) for k, v in r.items()]).to_csv(f"{out}/kalshi_minute_window.csv", index=False)
    f = lambda r, s, p: f"{r[(s, p)][0]:+.1f}c t{r[(s, p)][1]:.1f}"
    print(f"::notice::minute window :45-:59 minus rest, per city-day -- deep favourite fills dev {f(rb, 'favourite', 'dev')}, "
          f"holdout {f(rb, 'favourite', 'holdout')}; all deep fills holdout {f(rb, 'all', 'holdout')}; "
          f"touch favourite-side holdout {f(ra, 'favourite', 'holdout')}")


if __name__ == "__main__":
    main()
