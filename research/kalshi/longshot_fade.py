"""
research/kalshi/longshot_fade.py -- the longshot fade as a STRATEGY, not an average (Oct 18, 2026).

The scan (scan_analysis.py) showed cheap YES contracts settle below their price. That was contract-weighted over all
trades. This tests rules one could actually run:

  entry     SNAPSHOT at a fixed time before close (price = last trade then). A first-time-in-band rule was tried
            first and is biased: contracts that end up winning pass through every band on the way to $1, so
            "ever traded in band" over-selects winners (holdout: in-band markets settled YES 26% vs ~17c price).
            taker  buy NO at 1 - (p - half spread), taker fee
            maker  rest an offer at p; filled only if a later taker buys YES at >= p (adverse selection included)
  stake     $10 of NO per market; at most 3 markets per event (first three entries); held to settlement
  grid      YES band x window-to-close; dev = markets closing before 2026, holdout = 2026 (picked on dev, scored once)
  metrics   return per $ staked, t clustered by event, daily book (equal weight per close day): Sharpe (365),
            share of losing days, worst day; per-category on the dev pick

Limits: markets are the scan's volume-weighted sample (12 per series window), and only the most recent 3,000 trades per
market are cached, so for the busiest markets "first entry" may be later than a live bot's; fills assume the size was
there at that trade (our $10 is small next to typical trades).
"""
import glob, os
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
SC = f"{RD}/kalshi/scan"
OUT = f"{RD}/results"; os.makedirs(OUT, exist_ok=True)
BANDS = {"5-15": (.05, .15), "10-25": (.10, .25), "15-35": (.15, .35), "5-35": (.05, .35), "20-40": (.20, .40)}
HORIZONS = (1, 3, 6, 12, 24, 48, 96)          # hours before close at which the snapshot is taken
STALE_H = 24                                  # the last trade before the snapshot must be at most this old
HALF_SPREAD = (0.005, 0.01)                   # taker pays the YES price minus this (hits the bid), sensitivity
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
STAKE, PER_EVENT = 10.0, 3


def entries():
    """SNAPSHOT rule (no path selection): at close-h, price = last trade before then. If it's in the band:
    taker  buys NO at 1 - (p - half_spread), pays the taker fee;
    maker  rests a NO bid at 1 - p (i.e. offers YES at p); filled only if a later trade has a taker BUYING YES
           at >= p before close (adverse selection included), maker fee only where the series charges it.
    Markets whose cached trades (latest 3,000) start after the snapshot are skipped for that horizon."""
    S = pd.read_parquet(f"{SC}/series.parquet").set_index("ticker")
    rows = []
    for f in glob.glob(f"{SC}/s2_*.parquet"):
        if os.path.getsize(f) < 2000:
            continue
        t = pd.read_parquet(f, columns=["ticker", "t", "p", "c", "taker_side", "block", "series", "close_time", "R"])
        t = t[~t.block & (t.p > 0) & (t.p < 1)]
        if not len(t):
            continue
        t["ts"] = pd.to_datetime(t.t, utc=True, format="ISO8601")
        t = t.sort_values("ts")
        s = t.series.iloc[0]
        cat = S.category.get(s, "?"); maker_fee = "maker" in str(S.fee_type.get(s, ""))
        for tk, g in t.groupby("ticker", sort=False):
            close, R, t0 = g.close_time.iloc[0], g.R.iloc[0], g.ts.iloc[0]
            ts = g.ts.values; p = g.p.values
            for h in HORIZONS:
                snap = close - pd.Timedelta(hours=h)
                if t0 >= snap:
                    continue                                       # no cached trade before the snapshot
                i = np.searchsorted(ts, snap.to_datetime64(), side="left") - 1
                if i < 0 or (snap - g.ts.iloc[i]).total_seconds() > STALE_H * 3600:
                    continue
                ps = p[i]
                if not (0.05 <= ps < 0.40):
                    continue
                later = g.iloc[i + 1:]
                filled = bool(((later.taker_side == "yes") & (later.p >= ps - 1e-9)).any())
                rows.append((tk, s, cat, close, snap, h, ps, R, maker_fee, filled))
    E = pd.DataFrame(rows, columns=["ticker", "series", "cat", "close", "ts", "h", "p", "R", "maker_fee", "filled"])
    out = []
    for bn, (lo, hi) in BANDS.items():
        b = E[(E.p >= lo) & (E.p < hi)]
        for hs in HALF_SPREAD:
            x = b.copy(); x["var"] = f"taker_hs{hs * 100:.1f}c"; x["band"] = bn; x["px"] = x.p - hs
            out.append(x)
        x = b[b.filled].copy(); x["var"] = "maker"; x["band"] = bn; x["px"] = x.p
        out.append(x)
    E = pd.concat(out, ignore_index=True)
    E["win"] = E.h.astype(int).astype(str) + "h"
    return E


def pnl(E):
    N = STAKE / (1 - E.px)                                       # NO contracts bought with $10
    gross = N * (E.px - E.R)                                     # NO pays 1 if YES fails: (1-R) - (1-px)
    taker_fee = np.ceil(0.07 * N * E.px * (1 - E.px) * 100) / 100
    maker_fee = np.where(E.maker_fee, np.ceil(0.0175 * N * E.px * (1 - E.px) * 100) / 100, 0.0)
    E["ret"] = (gross - np.where(E["var"] == "maker", maker_fee, taker_fee)) / STAKE
    E["event"] = E.ticker.str.rsplit("-", n=1).str[0]
    E = E.sort_values("ts")
    E = E[E.groupby(["var", "band", "win", "event"]).cumcount() < PER_EVENT]
    return E


def stats(g):
    ev = g.groupby("event").ret.agg(["sum", "count"])
    m = g.ret.mean()
    se = np.sqrt(((ev["sum"] - m * ev["count"]) ** 2).sum()) / len(g) if len(ev) > 2 else np.nan
    d = g.groupby(pd.to_datetime(g.close).dt.date).ret.mean()
    return pd.Series(dict(markets=len(g), events=len(ev), win_pct=(g.R == 0).mean() * 100, ret_pct=m * 100,
                          t=m / se if se else np.nan, days=len(d),
                          day_sharpe=d.mean() / d.std() * np.sqrt(365) if len(d) > 5 else np.nan,
                          losing_days_pct=(d < 0).mean() * 100, worst_day_pct=d.min() * 100))


def main():
    E = pnl(entries())
    E.to_parquet(f"{OUT}/longshot_fade_entries.parquet", index=False)
    E["split"] = np.where(pd.to_datetime(E.close, utc=True) < SPLIT, "dev", "hold")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    G = E.groupby(["var", "band", "win", "split"]).apply(stats).unstack("split")
    G.columns = [f"{a}_{b}" for a, b in G.columns]
    cols = ["markets_dev", "ret_pct_dev", "t_dev", "day_sharpe_dev", "markets_hold", "ret_pct_hold", "t_hold", "day_sharpe_hold",
            "losing_days_pct_hold", "worst_day_pct_hold"]
    print("GRID (return per $ staked, %; t clustered by event)")
    print(G[cols].round(2).to_string())
    G.to_csv(f"{OUT}/longshot_fade_grid.csv")
    for var in ("taker_hs0.5c", "taker_hs1.0c", "maker"):
        g = G.loc[var]
        g = g[g.markets_dev >= 300]
        pick = g.t_dev.idxmax()
        r = g.loc[pick]
        print(f"\n{var} DEV PICK {pick}: dev {r.ret_pct_dev:+.2f}% t {r.t_dev:.1f} ({r.markets_dev:.0f} mkts) -> "
              f"HOLDOUT {r.ret_pct_hold:+.2f}% t {r.t_hold:.1f} ({r.markets_hold:.0f} mkts), daily Sharpe {r.day_sharpe_hold:.2f}, "
              f"losing days {r.losing_days_pct_hold:.0f}%, worst day {r.worst_day_pct_hold:.1f}%")
        sub = E[(E["var"] == var) & (E.band == pick[0]) & (E.win == pick[1])]
        c = sub.groupby(["cat", "split"]).apply(stats).unstack("split")
        c.columns = [f"{a}_{b}" for a, b in c.columns]
        print(c[[x for x in ("markets_dev", "ret_pct_dev", "t_dev", "markets_hold", "ret_pct_hold", "t_hold") if x in c]].round(2).to_string())
        print(f"::notice::longshot fade {var}: dev pick {pick[0]} YES, {pick[1]} to close: dev {r.ret_pct_dev:+.2f}%/trade t{r.t_dev:.1f}; "
              f"holdout {r.ret_pct_hold:+.2f}% t{r.t_hold:.1f}, daily Sharpe {r.day_sharpe_hold:.2f}")


if __name__ == "__main__":
    main()
