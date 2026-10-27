"""
research/kalshi/weather_other.py -- the rest of Kalshi's Climate & Weather category: is there room to put more money to
work? (Oct 27, 2026; Connor: ski resorts, climate, heatwaves, hourly temps, hurricanes, snow & rain, disasters)

Part 1  INVENTORY (live): every Climate & Weather series, how many markets are open now, their 24h volume and open
        interest, grouped by kind. This is the "how big is it" answer, and catches series newer than the Oct 17 scan
        (ski resorts weren't in it).
Part 2  THE BOT'S TEST on every non-daily-temperature series with >= 10 settled markets in the last year: weather_expand's
        deep-fill measurement (a quote 2c behind, filled by sweeps, up to 5 contracts, held to settlement), the 8 daily
        highs the bot trades as the control. Two windows: the bot's 6-48h, and any time >= 6h before close.
        Added for the scale question, because a cent earned on money locked up for a month is not a cent earned on
        money locked up for a day:
          fav_c          edge on fills where we'd hold the favourite (>= 50c) -- what the bot actually buys
          hold_days      median days from fill to close (capital locked up)
          ret_per_day    mean edge / (price held x days held): return on the money tied up, per day, in %
          per_mkt_day_$  the dollars a 5-lot quoter would have made per market per day of its life (capacity)
Public market data only. Output: results/kalshi_weather_other.csv, results/kalshi_weather_inventory.csv.
"""
import concurrent.futures as cf
import os, re, sys
import numpy as np, pandas as pd

os.environ.setdefault("SERIES_FILE", "weather_other")       # weather_expand: cache under kalshi/expand_other
os.environ.setdefault("N_PER", "40")
sys.path.insert(0, os.path.dirname(__file__))
import mm_scan as M
import weather_expand as W

RD = os.environ.get("RESEARCH_DATA", "research_data")
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 40)

KINDS = [("hourly temp", r"hourly"), ("ski / snowpack", r"\bski|resort|snowpack|powder"),
         ("daily low", r"^KXLOW|lowest|min temp"), ("daily high", r"^KXHIGH|highest|max(imum)? (daily )?temp"),
         ("weekly avg temp", r"^KXAVGT|average weekly"), ("snow", r"snow"), ("rain", r"rain|precip"),
         ("hurricane / tropical", r"hur|trop|cyclone|named storm"), ("heat", r"heat"),
         ("climate", r"climate|hottest|warmest|arctic|sea ice|co2|global|temperature (increase|deviation|anomaly)|el ni"),
         ("disaster", r"quake|tornado|fire|drought|flood|volcan|erupt|etna|fema|disaster|aqi|lake mead|river")]


def kind(ticker, title):
    s = f"{ticker} {title}".lower()
    for k, rx in KINDS:
        if re.search(rx, s, re.I):
            return k
    return "other"


def inventory(S):
    def one(s):
        mk = M.paged("/markets", "markets", max_pages=5, series_ticker=s, status="open")
        f = lambda m, *ks: next((M.f(m.get(k)) for k in ks if m.get(k) is not None), 0.0)
        return dict(series=s, open_markets=len(mk),
                    vol_24h=sum(f(m, "volume_24h_fp", "volume_24h") for m in mk),
                    open_interest=sum(f(m, "open_interest_fp", "open_interest") for m in mk),
                    next_close=min((m.get("close_time") for m in mk if m.get("close_time")), default=None))
    with cf.ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(one, S.ticker))
    I = pd.DataFrame(rows).merge(S.rename(columns={"ticker": "series"}), on="series")
    I["kind"] = [kind(a, b) for a, b in zip(I.series, I.title)]
    return I


def extra(s, t, win):
    sw, _ = W.deep_fills(t, *W.WINDOWS[win])
    if len(sw) < 20:
        return {}
    sw["held"] = np.where(sw.taker_side == "no", sw.px, 1 - sw.px)
    sw["days"] = np.maximum(sw.h, 1) / 24
    fav = sw[sw.held >= 0.5]
    f = lambda x: (x.e * x.c).sum() / x.c.sum() * 100 if len(x) else np.nan
    mk = t.drop_duplicates("ticker")
    life = ((mk.close - pd.to_datetime(t.groupby("ticker").t.min().reindex(mk.ticker).values, utc=True, format="ISO8601"))
            .dt.total_seconds() / 86400).clip(lower=1)
    return dict(fav_c=f(fav), fav_fills=len(fav), hold_days=float(sw.days.median()),
                ret_per_day=float((sw.e * sw.c).sum() / (sw.held * sw.days * sw.c).sum() * 100),
                per_mkt_day_usd=float((sw.e * sw.c).sum() / life.sum()))


def main():
    S = M.series_list()
    S = S[S.category == "Climate and Weather"].copy()
    print(f"Climate and Weather series: {len(S)}", flush=True)
    I = inventory(S)
    os.makedirs(f"{RD}/results", exist_ok=True)
    I.to_csv(f"{RD}/results/kalshi_weather_inventory.csv", index=False)
    act = I[I.open_markets > 0]
    print("\nINVENTORY, series with markets open now, by kind:")
    print(act.groupby("kind").agg(series=("series", "count"), open_markets=("open_markets", "sum"),
                                  vol_24h=("vol_24h", "sum"), open_interest=("open_interest", "sum"))
          .sort_values("vol_24h", ascending=False).round(0).to_string())
    print("\nnon-daily-temperature series open now (top 60 by 24h volume):")
    print(act[~act.kind.isin(["daily high", "daily low"])].sort_values("vol_24h", ascending=False)
          [["series", "kind", "title", "frequency", "open_markets", "vol_24h", "open_interest", "next_close"]].head(60).to_string(index=False))

    cand = list(S.ticker[~S.ticker.str.match(r"^KX(HIGH|LOW)")]) + sorted(W.CURRENT)
    rows = []
    with cf.ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(W.scan, s): s for s in cand}
        for fu in cf.as_completed(futs):
            s = futs[fu]
            try:
                t = fu.result()
                if t is None or not len(t) or "p" not in t:
                    continue
                for w in W.WINDOWS:
                    rows.append(dict(W.summarise(s, t, w), **extra(s, t, w)))
            except Exception as e:
                print(f"::warning::{s}: {type(e).__name__}: {str(e)[:150]}", flush=True)
    R = pd.DataFrame(rows).merge(I[["series", "title", "kind", "frequency", "open_markets", "vol_24h"]], on="series", how="left")
    R.to_csv(f"{RD}/results/kalshi_weather_other.csv", index=False)
    cols = ["series", "kind", "window", "frequency", "markets", "med_mkt_volume", "fills_per_mkt", "edge_c", "per_market_c", "t",
            "recent_c", "fav_c", "fav_fills", "hold_days", "ret_per_day", "per_mkt_day_usd", "open_markets", "vol_24h"]
    for w in W.WINDOWS:
        print(f"\nBOT'S TEST, window {w} (control = current daily highs first):")
        print(R[R.window == w].sort_values(["current", "per_market_c"], ascending=[False, False])[[c for c in cols if c in R]]
              .round(2).to_string(index=False))
    cur = R[R.current & (R.window == "6-48h")]
    good = R[~R.current & (R.per_market_c > 1.5) & (R.t > 2) & (R.recent_c > 0.5) & (R.fills_per_mkt >= 5)]
    print(f"::notice::weather other: control daily highs per-market {cur.per_market_c.median():.2f}c, return/day "
          f"{cur.ret_per_day.median():.2f}%; {len(good)} pass: "
          + ", ".join(f"{r.series}[{r.window}] {r.per_market_c:.1f}c t{r.t:.1f} {r.ret_per_day:.2f}%/d" for r in good.head(20).itertuples()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
