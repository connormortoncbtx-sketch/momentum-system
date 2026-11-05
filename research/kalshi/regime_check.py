"""
research/kalshi/regime_check.py -- did the bot's markets get more volatile in early November? (Nov 4, 2026)

Connor: the account chart went from "tabletops separated by sharp settling steps" to much more intraday movement.
If that is the weather/market changing (not the bot), the bot's own markets should show it in public data. Per
city-day of the bot's 11 series, market days Oct 20 - Nov 3:
  churn      sum over the city-day's buckets of |price moves| (cents) between consecutive trades, 6-30h before close
             -- how much the market repriced during the day
  surprise   the eventual winning bucket's price 12h before close (cents): low = the outcome surprised the market
  flips      how many times the favourite bucket (highest last price) changed, 6-30h before close
Averaged per market day across cities. Public data only -> results/kalshi_regime_check.csv.
"""
import concurrent.futures as cf
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import mm_scan as M

RD = os.environ.get("RESEARCH_DATA", "research_data")
SERIES = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHLAX", "KXHIGHMIA", "KXHIGHPHIL", "KXHIGHDEN", "KXHIGHAUS", "KXHIGHTDAL",
          "KXHIGHTATL", "KXHIGHTBOS", "KXHIGHTLV"]
D0, D1 = pd.Timestamp("2026-10-20"), pd.Timestamp("2026-11-03")
MON = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
pd.set_option("display.width", 200)


def mday(tk):
    d = tk.split("-")[1]
    return pd.Timestamp(2000 + int(d[:2]), MON[d[2:5]], int(d[5:7]))


def markets(s):
    rows = M.paged("/markets", "markets", max_pages=5, series_ticker=s, status="settled")
    out = []
    for m in rows:
        try:
            d = mday(m["ticker"])
        except Exception:
            continue
        if D0 <= d <= D1 and m.get("result") in ("yes", "no"):
            out.append(dict(series=s, ticker=m["ticker"], event=m.get("event_ticker"), day=d,
                            close=pd.Timestamp(m["close_time"]), R=m["result"] == "yes"))
    return out


def tape(m):
    tr = M.paged("/markets/trades", "trades", max_pages=20, ticker=m["ticker"],
                 min_ts=int((m["close"] - pd.Timedelta(hours=30)).timestamp()), max_ts=int((m["close"] - pd.Timedelta(hours=6)).timestamp()))
    t = pd.DataFrame([dict(ts=pd.Timestamp(x["created_time"]), p=M.f(x.get("yes_price_dollars"))) for x in tr])
    return m, (t.sort_values("ts") if len(t) else t)


def main():
    with cf.ThreadPoolExecutor(4) as ex:
        ms = [m for r in ex.map(markets, SERIES) for m in r]
    print(f"{len(ms)} markets", flush=True)
    with cf.ThreadPoolExecutor(4) as ex:
        tapes = list(ex.map(tape, ms))
    ev = {}
    for m, t in tapes:
        e = ev.setdefault(m["event"], dict(day=m["day"], series=m["series"], churn=0.0, surprise=np.nan, buckets={}))
        if len(t):
            e["churn"] += float(t.p.diff().abs().sum() * 100)
            s_ = t.set_index("ts").p; e["buckets"][m["ticker"]] = s_[~s_.index.duplicated(keep="last")]
            if m["R"]:
                p12 = t[t.ts <= m["close"] - pd.Timedelta(hours=12)].p
                e["surprise"] = float(p12.iloc[-1] * 100) if len(p12) else np.nan
    rows = []
    for k, e in ev.items():
        flips = 0
        if len(e["buckets"]) > 1:
            px = pd.DataFrame(e["buckets"]).sort_index().ffill()
            lead = px.idxmax(axis=1).dropna()
            flips = int((lead != lead.shift()).sum() - 1) if len(lead) else 0
        rows.append(dict(event=k, day=e["day"], series=e["series"], churn=e["churn"], surprise=e["surprise"], flips=max(flips, 0)))
    E = pd.DataFrame(rows)
    os.makedirs(f"{RD}/results", exist_ok=True)
    E.to_csv(f"{RD}/results/kalshi_regime_check.csv", index=False)
    D = E.groupby("day").agg(city_days=("event", "size"), churn_c=("churn", "mean"), winner_px_12h=("surprise", "mean"),
                             flips=("flips", "mean")).round(1)
    print(D.to_string())
    a, b = E[E.day <= "2026-10-30"], E[E.day >= "2026-10-31"]
    print(f"::notice::regime check -- Oct 20-30 vs Oct 31-Nov 3: churn {a.churn.mean():.0f}c -> {b.churn.mean():.0f}c per "
          f"city-day; winner's price 12h out {a.surprise.mean():.0f}c -> {b.surprise.mean():.0f}c; favourite flips "
          f"{a.flips.mean():.1f} -> {b.flips.mean():.1f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
