"""
research/kalshi/favorites_snapshot.py -- is buying "basically decided" 97-99.5c sides profitable? (Oct 21, 2026)

The trade-weighted table (fav side at 97/98/99c won 97.4/98.3/99.7% in 2025-26, +0.2 to +0.6c after fee) uses
the weighting that misled the longshot study. Here the honest version: at FIXED local times, take every side whose
ASK is 97-99.5c in the hourly candle at that hour (YES ask, or NO ask = 1 - YES bid), buy it, hold to settlement.
  fee   Kalshi taker fee ceil(0.07 * C * p * (1-p)) per order: C = 100 (big order) and C = 5 (a $5 leg)
  units per contract, and per city-day (all qualifying sides of one city-day averaged; t over city-days)
  split dev (< 2025-10-01) vs holdout; local times are the market day 10:00 ... 22:00 and the evening before 21:00
Also a chain check: at 14:00 local each day, the single best-priced side in 97-99c across cities, rolled with the
whole stake: survival and growth per leg.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import weather_edge as W

SPLIT = pd.Timestamp("2025-10-01")
SLOTS = {"eve-21": (-1, 21), "10": (0, 10), "12": (0, 12), "14": (0, 14), "16": (0, 16), "18": (0, 18), "20": (0, 20), "22": (0, 22)}


def fee(p, C):
    return np.ceil(0.07 * C * p * (1 - p) * 100 - 1e-9) / 100 / C


def main():
    m = W.market_intervals(pd.read_parquet(W.D / "markets.parquet"))
    m = m[m.y.isin([0, 1])][["ticker", "event_ticker", "station", "date", "y"]]
    cd = pd.read_parquet(W.D / "candles.parquet").dropna(subset=["bid", "ask"], how="all")
    cd["ts"] = pd.to_datetime(cd.ts, utc=True).astype("datetime64[ns, UTC]")
    cd = cd.sort_values("ts")
    rows = []
    for slot, (off, hr) in SLOTS.items():
        q = m.copy()
        q["ts"] = [(pd.Timestamp(d) + pd.Timedelta(days=off, hours=hr)).tz_localize(W.TZ[s]).tz_convert("UTC")
                   for d, s in zip(q.date, q.station)]
        q["ts"] = q.ts.astype("datetime64[ns, UTC]")
        j = pd.merge_asof(q.sort_values("ts"), cd[["ticker", "ts", "bid", "ask"]], on="ts", by="ticker",
                          direction="backward", tolerance=pd.Timedelta("90min"))
        for side in ("YES", "NO"):
            p = j.ask if side == "YES" else 1 - j.bid
            win = j.y if side == "YES" else 1 - j.y
            k = p.between(0.97, 0.995) & p.notna()
            x = j[k].assign(price=p[k], win=win[k], side=side, slot=slot)
            rows.append(x[["event_ticker", "ticker", "date", "slot", "side", "price", "win"]])
    T = pd.concat(rows, ignore_index=True)
    T["period"] = np.where(pd.to_datetime(T.date) < SPLIT, "dev", "hold")
    T["pnl100"] = T.win - T.price - fee(T.price, 100)
    T["pnl5"] = T.win - T.price - fee(T.price, 5)
    T["pb"] = pd.cut(T.price, [0.97, 0.98, 0.99, 0.996], right=False, labels=["97c", "98c", "99-99.5c"])
    os.makedirs(os.environ.get("RESEARCH_DATA", "research_data") + "/results", exist_ok=True)
    pd.set_option("display.width", 220)

    def st(g):
        pe = g.groupby("event_ticker").pnl100.mean()
        return pd.Series(dict(sides=len(g), city_days=len(pe), win_pct=g.win.mean() * 100, avg_price=g.price.mean() * 100,
                              pnl_big_c=g.pnl100.mean() * 100, pnl_5lot_c=g.pnl5.mean() * 100,
                              per_cityday_c=pe.mean() * 100, t=pe.mean() / pe.std() * np.sqrt(len(pe)) if len(pe) > 2 else np.nan,
                              losses=int((g.win == 0).sum())))
    print("BY PRICE (all snapshot times)")
    print(T.groupby(["period", "pb"], observed=True).apply(st).round(2).to_string())
    print("\nBY LOCAL TIME")
    print(T.groupby(["period", "slot"]).apply(st).round(2).to_string())
    # chain: 14:00 local, best price in 97-99c (most decided), one leg per calendar day
    C = T[(T.slot == "14") & (T.price <= 0.99)].sort_values("price").groupby("date").tail(1).sort_values("date")
    for per, g in C.groupby("period"):
        r = (g.win / g.price) * (1 - 0) - fee(g.price, 100) / g.price      # growth factor per leg (lose -> ~0)
        r = np.where(g.win == 1, 1 / g.price - fee(g.price, 100) / g.price, 0.0)
        surv = np.cumprod(r > 0)
        first_loss = int(np.argmax(r == 0)) + 1 if (r == 0).any() else None
        gm = np.exp(np.log(r[r > 0]).mean()) if (r > 0).any() else np.nan
        print(f"::notice::chain {per}: {len(g)} legs at 14:00 local, losses {(r == 0).sum()} ({(r == 0).mean() * 100:.1f}%), "
              f"first loss at leg {first_loss}, avg growth per winning leg {(gm - 1) * 100:.2f}%")
    T.to_csv(os.environ.get("RESEARCH_DATA", "research_data") + "/results/kalshi_favorites_snapshot.csv", index=False)
    for (per, pb), g in T.groupby(["period", "pb"], observed=True):
        s = st(g)
        print(f"::notice::{per} {pb}: {int(s.sides)} sides / {int(s.city_days)} city-days, won {s.win_pct:.2f}% at {s.avg_price:.2f}c, "
              f"net {s.pnl_big_c:+.2f}c (5-lot {s.pnl_5lot_c:+.2f}c), per city-day {s.per_cityday_c:+.2f}c t {s.t:.1f}")


if __name__ == "__main__":
    main()
