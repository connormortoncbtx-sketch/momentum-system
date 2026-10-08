"""
research/kalshi/fetch_kalshi.py
===============================
Data for testing whether a forecast-based model beats Kalshi daily-high
temperature markets. Runs in GitHub Actions; writes research_data/kalshi/.

  markets.parquet   every settled market in the US daily-high series below:
                    ticker, event_ticker, series, strike_type, floor/cap strike,
                    open/close time, result, volume
  candles.parquet   hourly candles per market (yes bid/ask close, last trade,
                    volume) from open to close -- live endpoint after Kalshi's
                    historical cutoff, /historical endpoint before it
  mos.parquet       GFS MOS ("GFS") and NBM text ("NBS") forecast max temps per
                    station from the Iowa Environmental Mesonet archive:
                    station, model, runtime (UTC), ftime (UTC), n_x

Env: ONLY=markets,candles,mos  SERIES=comma list  START=YYYY-MM-DD
"""
import concurrent.futures as cf
import datetime as dt
import io
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger("fetch_kalshi")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
API = "https://api.elections.kalshi.com/trade-api/v2"
UA = os.environ.get("SEC_USER_AGENT") or "MomentumAlpha research momentum-bot@users.noreply.github.com"

# series -> forecast station (closest official ASOS station to the contract's location)
SERIES = {
    "KXHIGHNY": "KNYC", "HIGHNY": "KNYC",
    "KXHIGHCHI": "KMDW", "HIGHCHI": "KMDW",
    "KXHIGHMIA": "KMIA", "HIGHMIA": "KMIA",
    "KXHIGHAUS": "KAUS", "HIGHAUS": "KAUS",
    "KXHIGHLAX": "KLAX", "HIGHLAX": "KLAX",
    "KXHIGHDEN": "KDEN", "KXDENHIGH": "KDEN", "HIGHDEN": "KDEN",
    "KXHIGHPHIL": "KPHL", "HIGHPHIL": "KPHL",
    "KXHIGHOU": "KHOU", "KXHOUHIGH": "KHOU",
    "KXHIGHTDC": "KDCA", "KXHIGHTBOS": "KBOS", "KXHIGHTDAL": "KDFW",
    "KXHIGHTPHX": "KPHX", "KXHIGHTSATX": "KSAT", "KXHIGHTSAN": "KSAN",
    "KXHIGHTKSAN": "KSAN", "KXHIGHTEWR": "KEWR",
}


class Limiter:
    def __init__(self, per_sec):
        self.gap, self.lock, self.t = 1.0 / per_sec, threading.Lock(), 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.t:
                time.sleep(self.t - now)
            self.t = max(now, self.t) + self.gap


KAL = Limiter(8)
IEM = Limiter(0.7)


def get_json(url, limiter=KAL, tries=5):
    for i in range(tries):
        limiter.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(3 * (i + 1)); continue
            log.warning(f"  HTTP {e.code} {url[:140]}")
            return None
        except Exception as e:
            time.sleep(3 * (i + 1))
    log.warning(f"  gave up: {url[:140]}")
    return None


def paged(path, params, key):
    out, cursor = [], None
    while True:
        q = dict(params, limit=1000 if key == "markets" else 200)
        if cursor:
            q["cursor"] = cursor
        d = get_json(f"{API}{path}?{urllib.parse.urlencode(q)}")
        if not d:
            break
        out.extend(d.get(key, []))
        cursor = d.get("cursor")
        if not cursor:
            break
    return out


def fetch_markets(series_list):
    rows = []
    for s in series_list:
        if not get_json(f"{API}/series/{s}"):
            log.info(f"  series {s}: not found"); continue
        live = paged("/markets", {"series_ticker": s, "status": "settled"}, "markets")
        hist = paged("/historical/markets", {"series_ticker": s}, "markets")
        if not hist:   # fall back to per-event lookup if the series filter isn't supported
            evs = paged("/events", {"series_ticker": s, "status": "settled"}, "events")
            seen = {m["event_ticker"] for m in live}
            todo = [e["event_ticker"] for e in evs if e["event_ticker"] not in seen]
            with cf.ThreadPoolExecutor(6) as ex:
                for res in ex.map(lambda et: paged("/historical/markets", {"event_ticker": et}, "markets"), todo):
                    hist.extend(res)
        ms = {m["ticker"]: m for m in hist + live}
        for m in ms.values():
            rows.append({
                "series": s, "station": SERIES[s], "ticker": m["ticker"], "event_ticker": m.get("event_ticker"),
                "strike_type": m.get("strike_type"), "floor_strike": m.get("floor_strike"),
                "cap_strike": m.get("cap_strike"), "open_time": m.get("open_time"),
                "close_time": m.get("close_time"), "result": m.get("result"),
                "volume": float(m.get("volume_fp") or m.get("volume") or 0),
                "title": m.get("title"), "settled_hist": m["ticker"] in {h["ticker"] for h in hist},
            })
        log.info(f"  series {s}: {len(ms):,} markets ({len(hist):,} historical)")
    df = pd.DataFrame(rows)
    for c in ("open_time", "close_time"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


def candles_for(m):
    s0 = int(m.open_time.timestamp()) - 3600
    s1 = int(m.close_time.timestamp()) + 3600
    q = urllib.parse.urlencode({"start_ts": s0, "end_ts": s1, "period_interval": 60})
    path = (f"/historical/markets/{m.ticker}/candlesticks" if m.settled_hist
            else f"/series/{m.series}/markets/{m.ticker}/candlesticks")
    d = get_json(f"{API}{path}?{q}")
    if not d and not m.settled_hist:
        d = get_json(f"{API}/historical/markets/{m.ticker}/candlesticks?{q}")
    out = []
    for c in (d or {}).get("candlesticks", []):
        f = lambda obj, k: float(obj[k]) if obj and obj.get(k) not in (None, "") else None
        yb, ya, pr = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
        out.append({"ticker": m.ticker, "ts": c["end_period_ts"],
                    "bid": f(yb, "close_dollars"), "ask": f(ya, "close_dollars"),
                    "last": f(pr, "close_dollars"), "vwap": f(pr, "mean_dollars"),
                    "volume": float(c.get("volume_fp") or 0)})
    return out


def _candle_frame(rows):
    df = pd.DataFrame(rows)
    if len(df):
        df["ts"] = pd.to_datetime(df.ts, unit="s", utc=True)
        for c in ("bid", "ask", "last", "vwap"):
            df[c] = df[c].astype("float32")
    return df


def fetch_candles(mk, path=None, budget_min=float(os.environ.get("BUDGET_MIN") or 290)):
    """Resumable: tickers already in `path` are skipped; checkpoints every 4,000
    markets and stops cleanly when the time budget runs out (rerun to continue)."""
    prev = pd.read_parquet(path) if path and Path(path).exists() else pd.DataFrame()
    if len(prev):
        mk = mk[~mk.ticker.isin(set(prev.ticker))]
    log.info(f"  candles: {len(prev):,} rows already, {len(mk):,} markets to fetch")
    t0, rows, done = time.monotonic(), [], 0
    todo = list(mk.itertuples(index=False))
    for i in range(0, len(todo), 4000):
        if (time.monotonic() - t0) / 60 > budget_min:
            log.info("  candles: time budget reached; rerun to continue"); break
        with cf.ThreadPoolExecutor(8) as ex:
            for res in ex.map(candles_for, todo[i:i + 4000]):
                rows.extend(res); done += 1
        log.info(f"  candles {done:,}/{len(todo):,} markets, {len(rows):,} rows")
        if path:
            pd.concat([prev, _candle_frame(rows)], ignore_index=True).to_parquet(path, index=False, compression="zstd")
    return pd.concat([prev, _candle_frame(rows)], ignore_index=True)


def _mos_frame(rows):
    df = pd.DataFrame(rows)
    if len(df):
        df["runtime"] = pd.to_datetime(df.runtime, utc=True)
        df["ftime"] = pd.to_datetime(df.ftime, utc=True)
    return df


def fetch_mos(stations, start, path=None):
    """Resumable: station/model pairs already in `path` are skipped, and the file
    is rewritten after each pair so a timeout keeps what was fetched."""
    prev = pd.read_parquet(path) if path and Path(path).exists() else pd.DataFrame()
    have = set(map(tuple, prev[["station", "model"]].drop_duplicates().values)) if len(prev) else set()
    rows = []
    months = pd.date_range(start, dt.date.today(), freq="MS")
    for st in sorted(stations):
        for model in ("GFS", "NBS"):
            if (st, model) in have:
                log.info(f"  MOS {st} {model}: already fetched"); continue
            n0 = len(rows)
            for m0 in months:
                m1 = m0 + pd.offsets.MonthBegin(1)
                q = urllib.parse.urlencode({"station": st, "model": model, "sts": f"{m0:%Y-%m-%d}T00:00Z",
                                            "ets": f"{m1:%Y-%m-%d}T00:00Z", "format": "csv"})
                txt = None
                for i in range(5):
                    IEM.wait()
                    try:
                        req = urllib.request.Request(f"https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?{q}",
                                                     headers={"User-Agent": UA})
                        with urllib.request.urlopen(req, timeout=90) as r:
                            txt = r.read().decode(); break
                    except urllib.error.HTTPError as e:
                        if e.code == 429:
                            time.sleep(10 * (i + 1)); continue
                        break
                    except Exception:
                        time.sleep(5 * (i + 1))
                if not txt or "runtime" not in txt[:200]:
                    continue
                d = pd.read_csv(io.StringIO(txt), usecols=lambda c: c in ("runtime", "ftime", "n_x", "x_n", "txn"))
                col = next((c for c in ("n_x", "x_n", "txn") if c in d.columns), None)
                if col is None:
                    continue
                d = d.dropna(subset=[col])
                for r in d.itertuples(index=False):
                    rows.append({"station": st, "model": model, "runtime": r.runtime, "ftime": r.ftime,
                                 "n_x": float(getattr(r, col))})
            log.info(f"  MOS {st} {model}: {len(rows) - n0:,} max/min rows")
            if path:
                pd.concat([prev, _mos_frame(rows)], ignore_index=True).to_parquet(path, index=False)
    return pd.concat([prev, _mos_frame(rows)], ignore_index=True)


def main():
    out = Path(os.environ.get("RESEARCH_DATA", "research_data")) / "kalshi"
    out.mkdir(parents=True, exist_ok=True)
    only = (os.environ.get("ONLY") or "markets,candles,mos").split(",")
    series = [s for s in (os.environ.get("SERIES") or ",".join(SERIES)).split(",") if s]
    start = os.environ.get("START") or "2021-06-01"
    if "markets" in only:
        mk = fetch_markets(series)
        mk.to_parquet(out / "markets.parquet", index=False)
    else:
        mk = pd.read_parquet(out / "markets.parquet")
    log.info(f"markets: {len(mk):,} across {mk.series.nunique()} series, "
             f"{mk.open_time.min():%Y-%m-%d}..{mk.close_time.max():%Y-%m-%d}")
    if "candles" in only:
        cd = fetch_candles(mk[mk.open_time >= pd.Timestamp(start, tz="UTC")], out / "candles.parquet")
        log.info(f"candles: {len(cd):,} rows for {cd.ticker.nunique() if len(cd) else 0:,} markets")
    if "mos" in only:
        mos = fetch_mos(set(mk.station), start, out / "mos.parquet")
        mos.to_parquet(out / "mos.parquet", index=False)
        log.info(f"mos: {len(mos):,} rows, {mos.station.nunique() if len(mos) else 0} stations")
    summary = {"markets": len(mk), "series": sorted(mk.series.unique().tolist())}
    (out / "README.md").write_text(json.dumps(summary, indent=1))
    print(f"::notice::kalshi data: {len(mk)} markets, series {summary['series']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
