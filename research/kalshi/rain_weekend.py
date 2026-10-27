"""
research/kalshi/rain_weekend.py -- weekend rain (KXRAINWKND), the full history, cut every way.   (Oct 27, 2026)

weather_other.py sampled 39 markets: +11.7c per contract in the bot's 6-48h window, t 2.3. This takes EVERY settled
market (the series began Aug 2026: one yes/no market per city per weekend) and the full tape of each, and replays the
bot's backtest (a quote 2c behind, filled by sweeps that trade 2c past their first print, up to 5 contracts, held to
settlement; favourite = the side held is >= 50c; quiet window :45-:59 skipped as the bot does).

The honest error bar: every city's market on one weekend rides the same weather systems, so the independent unit is
closer to the WEEKEND than the market. Both are shown (2 s.e. clustered by market, and by weekend), plus leave-one-
weekend-out: the edge recomputed with each weekend dropped, min / max -- a result that hangs on one storm shows here.
Cuts: window (6-48h = the bot; 6h+ = any time), favourite vs long shot, hours before close, price band, city, weekend.
Capacity: backtest contracts per market and dollars per weekend across all cities.
Public data only -> results/kalshi_rain_weekend.csv.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import mm_scan as M

RD = os.environ.get("RESEARCH_DATA", "research_data")
SERIES = os.environ.get("SERIES", "KXRAINWKND")
OUT = f"{RD}/kalshi/rain_weekend"; os.makedirs(OUT, exist_ok=True)
OFF, SIZE = 0.02, 5
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)


def markets():
    rows = M.paged("/markets", "markets", max_pages=20, series_ticker=SERIES, status="settled") + \
           M.paged("/historical/markets", "markets", max_pages=20, series_ticker=SERIES)
    d = pd.DataFrame([dict(ticker=m["ticker"], event=m.get("event_ticker"), close=m.get("close_time"), result=m.get("result"),
                           volume=M.f(m.get("volume_fp", m.get("volume")))) for m in rows if m.get("result") in ("yes", "no")])
    d = d.drop_duplicates("ticker")
    d["close"] = pd.to_datetime(d.close, utc=True)
    return d


def trades(tk):
    p = f"{OUT}/{tk}.parquet"
    if os.path.exists(p):
        return pd.read_parquet(p)
    rows = []
    for path in ("/markets/trades", "/historical/trades"):
        rows = M.paged(path, "trades", max_pages=40, ticker=tk)
        if rows:
            break
    t = pd.DataFrame([dict(ticker=tk, t=r.get("created_time"), p=M.f(r.get("yes_price_dollars")),
                           c=M.f(r.get("count_fp", r.get("count"))), side=r.get("taker_side"),
                           block=bool(r.get("is_block_trade"))) for r in rows])
    t.to_parquet(p, index=False)
    return t


def sim(t, mk):
    """Backtest fills for one market (the bot's rule; a sweep's first print = the one nearest the touch)."""
    t = t[(t.p > 0) & (t.p < 1) & ~t.block].copy()
    if t.empty:
        return t
    t["ts"] = pd.to_datetime(t.t, utc=True, format="ISO8601")
    g = t.groupby(["ts", "side"])
    sw = g.p.agg(["max", "min"]).reset_index()
    sw["first"] = np.where(sw.side == "no", sw["max"], sw["min"])
    t = t.merge(sw[["ts", "side", "first"]], on=["ts", "side"])
    t["q"] = np.where(t.side == "no", np.where(t.p <= t["first"] - OFF + 1e-9, t.c, 0), np.where(t.p >= t["first"] + OFF - 1e-9, t.c, 0))
    f = t.groupby(["ts", "side"]).agg(first=("first", "first"), q=("q", "sum")).reset_index()
    f = f[f.q > 0]
    f["d"] = np.where(f.side == "no", 1, -1)
    f["px"] = np.where(f.d > 0, f["first"] - OFF, f["first"] + OFF)
    f = f[(f.px >= 0.03) & (f.px <= 0.97)]
    f["held"] = np.where(f.d > 0, f.px, 1 - f.px)
    f["c"] = np.minimum(f.q, SIZE)
    R = 1.0 if mk.result == "yes" else 0.0
    f["e"] = f.d * (R - f.px) * 100                    # cents per contract
    f["h"] = (mk.close - f.ts).dt.total_seconds() / 3600
    f["ticker"], f["event"], f["city"] = mk.ticker, mk.event, mk.ticker.rsplit("-", 1)[1]
    f["quiet"] = f.ts.dt.minute.between(45, 59)
    return f


def stat(x, unit):
    if not len(x):
        return dict(contracts=0)
    g = x.assign(v=x.e * x.c).groupby(unit).agg(v=("v", "sum"), w=("c", "sum"))
    N = g.w.sum(); m = g.v.sum() / N; G = len(g)
    se = np.sqrt(((g.v - m * g.w) ** 2).sum() / N ** 2 * G / (G - 1)) if G > 2 else np.nan
    return dict(contracts=N, units=G, edge_c=m, se2=2 * se)


def row(label, x):
    a, b = stat(x, "ticker"), stat(x, "event")
    lo = []
    for ev in x.event.unique():
        y = x[x.event != ev]
        if len(y):
            lo.append((y.e * y.c).sum() / y.c.sum())
    return dict(cut=label, contracts=a.get("contracts", 0), markets=a.get("units", 0), weekends=b.get("units", 0),
                edge_c=a.get("edge_c", np.nan), se_mkt=a.get("se2", np.nan), se_wknd=b.get("se2", np.nan),
                t_wknd=a.get("edge_c", np.nan) / (b.get("se2", np.nan) / 2) if b.get("se2") else np.nan,
                lowo_min=min(lo) if lo else np.nan, lowo_max=max(lo) if lo else np.nan)


def main():
    mk = markets()
    print(f"{SERIES}: {len(mk)} settled markets, {mk.event.nunique()} weekends, {mk.ticker.str.rsplit('-', n=1).str[1].nunique()} cities, "
          f"{mk.close.min():%Y-%m-%d} .. {mk.close.max():%Y-%m-%d}; YES rate {(mk.result == 'yes').mean():.0%}", flush=True)
    F = []
    for i, m in enumerate(mk.itertuples()):
        try:
            F.append(sim(trades(m.ticker), m))
        except Exception as e:
            print(f"::warning::{m.ticker}: {type(e).__name__}: {str(e)[:120]}")
        if i % 50 == 49:
            print(f"  {i + 1}/{len(mk)}", flush=True)
    F = pd.concat([f for f in F if len(f)], ignore_index=True)
    F = F[F.h >= 6]
    bot = F[(F.h <= 48) & (F.held >= 0.5) & ~F.quiet]
    rows = [row("BOT: 6-48h, favourite, not quiet", bot),
            row("6-48h, all prices", F[F.h <= 48]),
            row("6h+, favourite", F[F.held >= 0.5]),
            row("6h+, all prices", F),
            row("6-48h long shot (<50c)", F[(F.h <= 48) & (F.held < 0.5)])]
    for lo_, hi in ((6, 24), (24, 48), (48, 96), (96, 1e9)):
        rows.append(row(f"hours before close {lo_}-{hi if hi < 1e9 else '+'} (fav)", F[(F.h >= lo_) & (F.h < hi) & (F.held >= 0.5)]))
    for lo_, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.98)):
        rows.append(row(f"held {lo_:.0%}-{hi:.0%} (bot window)", bot[(bot.held >= lo_) & (bot.held < hi)]))
    Rt = pd.DataFrame(rows)
    cities = pd.DataFrame([dict(row(c, bot[bot.city == c]), cut=c) for c in sorted(bot.city.unique())]).sort_values("edge_c")
    wk = pd.DataFrame([dict(row(e, bot[bot.event == e]), cut=e) for e in sorted(bot.event.unique())])
    os.makedirs(f"{RD}/results", exist_ok=True)
    pd.concat([Rt.assign(table="cut"), cities.assign(table="city"), wk.assign(table="weekend")]).to_csv(
        f"{RD}/results/kalshi_rain_weekend.csv", index=False)
    print("\nCUTS (edge c/contract; ±2se clustered by market and by weekend; leave-one-weekend-out min/max):")
    print(Rt.round(2).to_string(index=False))
    print("\nBY CITY (bot window):"); print(cities.round(2).to_string(index=False))
    print("\nBY WEEKEND (bot window):"); print(wk.round(2).to_string(index=False))
    per_wk = (bot.e * bot.c).groupby(bot.event).sum() / 100
    cpm = bot.groupby("ticker").c.sum().reindex(mk.ticker).fillna(0)
    print(f"\ncapacity (bot window): {cpm.mean():.1f} contracts per market (median {cpm.median():.0f}); "
          f"backtest $/weekend all cities: mean {per_wk.mean():+.2f}, median {per_wk.median():+.2f}")
    b = Rt.iloc[0]
    print(f"::notice::{SERIES} full history: bot backtest {b.edge_c:+.1f}c/contract on {b.contracts:.0f} contracts, "
          f"{b.markets} markets / {b.weekends} weekends; ±{b.se_mkt:.1f} by market, ±{b.se_wknd:.1f} by weekend "
          f"(t {b.t_wknd:.1f}); leave-one-weekend-out {b.lowo_min:+.1f}..{b.lowo_max:+.1f}; {cpm.mean():.1f} contracts/market, "
          f"${per_wk.mean():.2f}/weekend")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
