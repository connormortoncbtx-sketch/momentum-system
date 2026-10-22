"""
research/kalshi/fav_curve.py -- how finely should favourite-only quoting be tuned? (Oct 22, 2026)

The live bot quotes only the side priced >= FAV_MIN = 50c, from 4 crude price bands. Here, on the bot's simulated deep
fills (profit_lock.fills: sweep reaching 2c behind, 5-lot, 6-48h, 25/market cap) held to settlement:
  1. edge per contract by held-side price at 2c resolution, dev vs holdout -- the shape of the curve
  2. cut-off sweep 20..84c: P&L and per-city-day t in each period; cut-off chosen on DEV only, scored on holdout
  3. finer conditioning: edge by price band x hours to close (does the right cut-off move with time to close?)
  4. sizing by edge: contracts scaled by the DEV edge of the 10c band (0 where dev edge <= 0), vs flat 5-lot,
     scored on holdout per city-day mean / sd (a Sharpe-like ratio) and total P&L per contract of capital
dev < 2025-10-01 <= holdout. t from per-city-day sums (fills in one city-day are one bet).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from mm_perfill import load, SPLIT
from profit_lock import fills

pd.set_option("display.width", 240)


def cityday_t(g, w=None):
    w = g.c if w is None else w
    ev = g.ticker.str.rsplit("-", n=1).str[0]
    s = (g.e * w).groupby(ev).sum()
    return s.sum(), s.mean() / s.std() * np.sqrt(len(s)) if len(s) > 2 and s.std() > 0 else np.nan, s


def main():
    t = load()
    F = fills(t)
    F["c"] = F.q.abs()
    F["e"] = np.where(F.q > 0, F.R - F.px, F.px - F.R)          # per contract, held to settlement
    F["hp"] = np.where(F.q > 0, F.px, 1 - F.px)                 # price paid for the side held
    F["period"] = np.where(F.t < SPLIT, "dev", "hold")
    print(f"fills {len(F):,}, contracts {F.c.sum():,.0f}; dev {(F.period == 'dev').sum():,} hold {(F.period == 'hold').sum():,}")

    # 1. curve
    F["b2"] = (np.floor(F.hp * 50) / 50).round(2)
    cur = F.groupby(["b2", "period"]).apply(lambda g: pd.Series(dict(
        contracts=g.c.sum(), edge_c=(g.e * g.c).sum() / g.c.sum() * 100,
        se_c=np.sqrt(((g.e - (g.e * g.c).sum() / g.c.sum()) ** 2 * g.c).sum() / g.c.sum() / max(g.ticker.str.rsplit("-", n=1).str[0].nunique(), 1)) * 100))).unstack("period")
    print("\n1. EDGE PER CONTRACT BY HELD-SIDE PRICE (2c bins), cents  [se ~ per city-day]")
    print(cur.round(2).to_string())

    # 2. cut-off sweep
    rows = []
    for cut in np.arange(0.20, 0.86, 0.02):
        r = dict(cut_c=round(cut * 100))
        for per, g in F.groupby("period"):
            k = g[g.hp >= cut - 1e-9]
            tot, tt, _ = cityday_t(k)
            r[f"{per}_pnl_usd"] = tot
            r[f"{per}_edge_c"] = (k.e * k.c).sum() / k.c.sum() * 100
            r[f"{per}_t"] = tt
            r[f"{per}_contracts"] = k.c.sum()
        rows.append(r)
    S = pd.DataFrame(rows)
    best = S.loc[S.dev_pnl_usd.idxmax(), "cut_c"]
    print("\n2. CUT-OFF SWEEP (quote only sides priced >= cut)")
    print(S.round(2).to_string(index=False))
    b, c50 = S[S.cut_c == best].iloc[0], S[S.cut_c == 50].iloc[0]
    print(f"   dev-best cut-off {best:.0f}c: holdout P&L ${b.hold_pnl_usd:,.0f} (t {b.hold_t:.1f}) vs 50c ${c50.hold_pnl_usd:,.0f} (t {c50.hold_t:.1f})")

    # 3. price x hours
    F["pb"] = pd.cut(F.hp, [0, .3, .4, .5, .6, .7, .8, .9, 1], labels=["<30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90+"])
    F["hb"] = pd.cut((pd.to_datetime(F.close, utc=True) - F.t).dt.total_seconds() / 3600, [6, 12, 24, 48], labels=["6-12h", "12-24h", "24-48h"])
    X = F.groupby(["period", "hb", "pb"], observed=True).apply(lambda g: (g.e * g.c).sum() / g.c.sum() * 100).unstack("pb")
    print("\n3. EDGE BY PRICE BAND x HOURS TO CLOSE, cents per contract")
    print(X.round(1).to_string())

    # 4. sizing by dev edge per 10c band
    F["b10"] = (np.floor(F.hp * 10) / 10).round(1)
    dev_edge = F[F.period == "dev"].groupby("b10").apply(lambda g: (g.e * g.c).sum() / g.c.sum())
    scale = (dev_edge.clip(lower=0) / dev_edge.clip(lower=0).max()).rename("scale")
    print("\n4. SIZING BY DEV EDGE (10c bands): dev edge c / size multiplier")
    print(pd.concat([dev_edge * 100, scale], axis=1).round(2).to_string())
    H = F[F.period == "hold"].copy()
    H["w_flat50"] = np.where(H.hp >= 0.5, H.c, 0.0)
    H["w_dev50"] = np.where(H.hp >= 0.5, H.c, 0.0) * 0 + np.where(H.hp >= best / 100, H.c, 0.0)
    H["w_scaled"] = H.c * H.b10.map(scale).fillna(0) * 2          # x2 so peak bands trade up to 10 (a 2x bigger max)
    sz = []
    print("\n   HOLDOUT, per city-day:            P&L $    mean $   sd $   mean/sd   $ per contract   contracts")
    for name in ("w_flat50", "w_dev50", "w_scaled"):
        tot, tt, s = cityday_t(H, H[name])
        cap = (H[name] * H.hp).sum()
        sz.append(dict(rule=name, pnl=tot, mean=s.mean(), sd=s.std(), contracts=H[name].sum()))
        print(f"   {name:30} {tot:9,.0f} {s.mean():8.3f} {s.std():7.3f} {s.mean() / s.std():8.3f} {tot / H[name].sum() * 100:+12.2f}c {H[name].sum():11,.0f}")
    out = os.path.join(os.environ.get("RESEARCH_DATA", "research_data"), "results")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(sz).to_csv(f"{out}/kalshi_fav_sizing_holdout.csv", index=False)
    cur.to_csv(f"{out}/kalshi_fav_curve.csv"); S.to_csv(f"{out}/kalshi_fav_cutoffs.csv", index=False)
    X.to_csv(f"{out}/kalshi_fav_price_hours.csv"); pd.concat([dev_edge * 100, scale], axis=1).to_csv(f"{out}/kalshi_fav_sizing.csv")
    tot50, t50, s50 = cityday_t(H, H.w_flat50); totS, tS, sS = cityday_t(H, H.w_scaled); totB, tB, _ = cityday_t(H, H.w_dev50)
    print(f"::notice::fav cut-off: dev-best {best:.0f}c -> holdout ${totB:,.0f} (t {tB:.1f}) vs 50c ${tot50:,.0f} (t {t50:.1f}); "
          f"edge-scaled sizing holdout mean/sd {sS.mean() / sS.std():.3f} vs flat {s50.mean() / s50.std():.3f}")


if __name__ == "__main__":
    main()
