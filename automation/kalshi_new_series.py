"""
automation/kalshi_new_series.py -- weekly watch for NEW Kalshi series the market-making bot could fold in.

Each run (weekly, .github/workflows/kalshi_watch.yml):
  1. list every Kalshi series; anything not in data/kalshi_series_seen.json is NEW (first run = baseline, no alerts)
  2. for new series: category, frequency, fee type, open markets, 24h volume, median quoted spread -> watchlist
     (data/kalshi_watchlist.json), skipping one-off / sports / politics / elections series and anything with no volume
  3. every watched series is re-checked weekly; once it has >= 20 settled markets it "graduates": the bot's own
     deep-fill test (research/kalshi/weather_expand.py: quote 2c behind, size 5, held to settlement) runs on it,
     and the result (per-fill and per-market edge, t, fills per market) is recorded and reported
  4. a short summary goes to the phone via ntfy (NTFY_CHANNEL)
KEYWORD ALERT (daily, MODE=keywords): any new series whose title mentions company dividends / earnings / buybacks /
guidance / payouts / splits / revenue (not "mention"/"say" earnings-call markets) is pushed to the phone at once,
with its open markets and quotes -- new, thin corporate-event markets are where mispriced near-certainties can sit.
Already-alerted series are kept in data/kalshi_keyword_alerted.json.
Adding a series to the live bot stays a human decision (SERIES in /etc/kmm/env). Watched series expire after 180 days.
Why: new series showed somewhat higher maker edge in their first month (+1.1c per contract, +0.25c per market, t 1.9
-- weak), so the point is early notice plus a real test as soon as there's enough history, not automatic trading.
"""
import datetime as dt
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research" / "kalshi"))
os.environ.setdefault("RESEARCH_DATA", str(ROOT / "research_data"))
import mm_scan as M            # noqa: E402  (get / paged / f, rate-limited public API client)

DATA = ROOT / "data"
SEEN, WATCH = DATA / "kalshi_series_seen.json", DATA / "kalshi_watchlist.json"
ALERTED = DATA / "kalshi_keyword_alerted.json"
KEYWORDS = re.compile(r"\b(dividends?|earnings|EPS|buybacks?|repurchases?|guidance|payouts?|stock splits?|revenue)\b", re.I)
NOT_KEYWORDS = re.compile(r"\b(say|says|mention|mentions|tax|tariff|DOGE|stimulus)\b", re.I)
SKIP_CATS = {"Sports", "Politics", "Elections"}
RECURRING = {"daily", "weekly", "monthly", "custom", "hourly"}
GRADUATE_AT, EXPIRE_DAYS = 20, 180
WX_ALERTED = DATA / "kalshi_weather_alerted.json"
WEATHER = re.compile(r"^KX(HIGH|LOW)")
# Oct 30, 2026: likely next daily-temperature cities and the station Kalshi would most plausibly settle on (NWS climate
# report). A FALLBACK hint only -- the alert quotes the market's own rules, which are what count. "!" = a city with more
# than one airport / a settlement station people commonly get wrong.
LIKELY_STATIONS = {
    "PDX": "KPDX Portland Intl", "SLC": "KSLC Salt Lake City Intl", "BNA": "KBNA Nashville Intl",
    "CLT": "KCLT Charlotte Douglas", "DTW": "KDTW Detroit Metro", "MCI": "KMCI Kansas City Intl ! (not downtown KMKC)",
    "STL": "KSTL St. Louis Lambert", "TPA": "KTPA Tampa Intl", "MCO": "KMCO Orlando Intl ! (not Executive KORL)",
    "IND": "KIND Indianapolis Intl", "CMH": "KCMH Columbus", "CLE": "KCLE Cleveland Hopkins", "PIT": "KPIT Pittsburgh Intl",
    "BWI": "KBWI Baltimore-Washington (DC itself settles on KDCA)", "RDU": "KRDU Raleigh-Durham",
    "SAC": "KSAC Sacramento Executive ! (or KSMF Intl -- check)", "ABQ": "KABQ Albuquerque", "TUS": "KTUS Tucson",
    "MEM": "KMEM Memphis", "JAX": "KJAX Jacksonville", "HNL": "PHNL Honolulu", "ANC": "PANC Anchorage",
    "SJC": "KSJC San Jose", "BUF": "KBUF Buffalo", "MKE": "KMKE Milwaukee", "OMA": "KOMA Omaha", "RIC": "KRIC Richmond",
    "BOI": "KBOI Boise", "ELP": "KELP El Paso", "IAH": "KIAH Houston Bush ! (Kalshi's Houston daily high uses Hobby KHOU)",
    "ORD": "KORD O'Hare ! (Kalshi's Chicago uses Midway KMDW)", "IAD": "KIAD Dulles ! (Kalshi's DC uses National KDCA)",
}
STATION_RX = re.compile(r"\b[KPC][A-Z]{3}\b|(?i:Central Park|Midway|Hobby)|[A-Z][a-z]+(?: [A-Z][a-z]+)* (?:International )?Airport")
TODAY = dt.date.today().isoformat()


def all_series():
    s = pd.DataFrame(M.paged("/series", "series", max_pages=100))
    for c in ("category", "frequency", "fee_type", "title"):
        if c not in s:
            s[c] = ""
    return s


def open_stats(series):
    mk = M.paged("/markets", "markets", max_pages=5, series_ticker=series, status="open")
    if not mk:
        return dict(open_markets=0, vol24h=0.0, med_spread_c=None)
    bid = np.array([M.f(m.get("yes_bid_dollars")) for m in mk]); ask = np.array([M.f(m.get("yes_ask_dollars")) for m in mk])
    ok = (bid > 0) & (ask > 0) & (ask > bid)
    return dict(open_markets=len(mk), vol24h=float(np.nansum([M.f(m.get("volume_24h_fp", m.get("volume_24h"))) for m in mk])),
                med_spread_c=round(float(np.median(ask[ok] - bid[ok]) * 100), 1) if ok.any() else None)


def settled_count(series):
    return len(M.paged("/markets", "markets", max_pages=3, series_ticker=series, status="settled"))


def graduate(series):
    import weather_expand as X
    t = X.scan(series)
    if t is None or not len(t) or "p" not in t:
        return {"result": "no trade data"}
    out = {}
    for w in X.WINDOWS:
        r = X.summarise(series, t, w)
        out[w] = {k: (round(float(v), 3) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
                  for k, v in r.items() if k in ("markets", "fills", "fills_per_mkt", "edge_c", "per_market_c", "t", "recent_c")}
    best = max(out.values(), key=lambda r: r.get("per_market_c") or -99)
    passed = (best.get("per_market_c") or 0) > 1.5 and (best.get("t") or 0) > 2.5 and (best.get("fills_per_mkt") or 0) >= 5
    return {"windows": out, "passes": bool(passed)}


def notify(msg, title="Kalshi new-series watch", urgent=False):
    ch = os.environ.get("NTFY_CHANNEL")
    print(msg)
    if ch:
        h = {"Title": title}
        if urgent:
            h.update({"Priority": "high", "Tags": "rotating_light"})
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{ch}", data=msg.encode()[:3500],
                                                          headers=h), timeout=20)
        except Exception as e:
            print(f"ntfy: {e}")


def keyword_check(S, seen):
    """Alert once for each new series whose title is about a company dividend / earnings / buyback event."""
    if seen is None:
        return
    done = set(json.loads(ALERTED.read_text())) if ALERTED.exists() else set()
    new = S[~S.ticker.isin(seen) & ~S.ticker.isin(done)]
    t = new.title.fillna("")
    meta = new.category.fillna("") + " " + new.ticker.fillna("")          # "Mentions" category / ...MENTION... tickers
    hits = new[t.str.contains(KEYWORDS) & ~t.str.contains(NOT_KEYWORDS) & ~meta.str.contains(r"mention", case=False)]
    for r in hits.itertuples():
        try:
            mk = M.paged("/markets", "markets", max_pages=2, series_ticker=r.ticker, status="open")
        except Exception as e:
            mk = []; print(f"markets {r.ticker}: {e}")
        lines = [f"NEW corporate-event series {r.ticker} ({r.category}, {r.frequency}): {r.title}",
                 f"{len(mk)} open markets"]
        for m in mk[:6]:
            lines.append(f"- {m.get('title', m.get('ticker'))[:70]} | yes bid {M.f(m.get('yes_bid_dollars'))} "
                         f"ask {M.f(m.get('yes_ask_dollars'))} | vol {M.f(m.get('volume_fp', m.get('volume')))} | "
                         f"closes {str(m.get('close_time'))[:10]}")
        lines.append("Check: market price vs the company's history; buy only if the gap beats fees + the chance of a surprise.")
        notify("\n".join(lines), title="Kalshi: new dividend/earnings-type market", urgent=True)
        done.add(r.ticker)
    ALERTED.write_text(json.dumps(sorted(done)))
    print(f"keyword check: {len(new)} new series, {len(hits)} corporate-event hits")


def weather_check(S, seen):
    """Daily: alert once for each NEW daily-temperature series (KXHIGH*/KXLOW*), with its settlement station from the
    market's own rules and the one-line command to add it to the bot at the reduced starter cap."""
    if seen is None:
        return
    done = set(json.loads(WX_ALERTED.read_text())) if WX_ALERTED.exists() else set()
    new = S[~S.ticker.isin(seen) & ~S.ticker.isin(done) & S.ticker.fillna("").str.match(WEATHER)]
    for r in new.itertuples():
        try:
            mk = M.paged("/markets", "markets", max_pages=2, series_ticker=r.ticker, status="open")
        except Exception as e:
            mk = []; print(f"markets {r.ticker}: {e}")
        rules = " ".join(str(mk[0].get(k) or "") for k in ("rules_primary", "rules_secondary")) if mk else ""
        hits = sorted({m.group(0) for m in STATION_RX.finditer(rules)})[:4]
        code = re.sub(r"^KX(HIGH|LOW)T?", "", r.ticker)
        hint = LIKELY_STATIONS.get(code[:3])
        lines = [f"NEW weather series {r.ticker}: {r.title} ({r.frequency})", f"{len(mk)} open markets"]
        lines.append(f"Settles on (from the rules): {', '.join(hits) if hits else 'not found -- read the market rules'}"
                     + (f" | likely: {hint}" if hint else ""))
        if rules:
            lines.append("Rules: " + rules[:300])
        if r.ticker.startswith("KXHIGH"):
            lines.append("Starter add (reduced city-day cap until the auto-test passes at 20 settled markets), on the server:")
            lines.append(f"  sudo kmm add-series {r.ticker}")
        notify("\n".join(lines), title="Kalshi: NEW weather market", urgent=True)
        done.add(r.ticker)
    WX_ALERTED.write_text(json.dumps(sorted(done)))
    print(f"weather check: {len(new)} new temperature series")


def main():
    S = all_series()
    seen = set(json.loads(SEEN.read_text())) if SEEN.exists() else None
    keyword_check(S, seen)
    weather_check(S, seen)
    if os.environ.get("MODE") == "keywords":          # daily: alert only, leave the weekly report its new series
        return
    watch = json.loads(WATCH.read_text()) if WATCH.exists() else {}
    now = set(S.ticker)
    if seen is None:
        SEEN.write_text(json.dumps(sorted(now)))
        WATCH.write_text(json.dumps(watch, indent=1))
        notify(f"Kalshi watch baseline: {len(now):,} series recorded; new ones will be reported from next week.")
        return
    new = S[~S.ticker.isin(seen)]
    added = []
    for r in new.itertuples():
        if r.category in SKIP_CATS or str(r.frequency) not in RECURRING:
            continue
        st = open_stats(r.ticker)
        if st["open_markets"] == 0 and st["vol24h"] == 0:
            continue
        watch[r.ticker] = dict(title=r.title, category=r.category, frequency=r.frequency, fee_type=r.fee_type,
                               first_seen=TODAY, checks=[{**st, "date": TODAY}])
        added.append((r.ticker, r.title, r.category, r.frequency, st))
    graduated, expired = [], []
    for tk, w in list(watch.items()):
        if (dt.date.today() - dt.date.fromisoformat(w["first_seen"])).days > EXPIRE_DAYS and "test" not in w:
            expired.append(tk); watch.pop(tk); continue
        if tk in [a[0] for a in added] or "test" in w:
            continue
        try:
            st = open_stats(tk); n = settled_count(tk)
        except M.Fail:
            continue
        w["checks"].append({**st, "settled": n, "date": TODAY})
        w["checks"] = w["checks"][-12:]
        if n >= GRADUATE_AT:
            w["test"] = {"date": TODAY, **graduate(tk)}
            graduated.append((tk, w))
    SEEN.write_text(json.dumps(sorted(seen | now)))
    WATCH.write_text(json.dumps(watch, indent=1))

    lines = [f"{len(new)} new Kalshi series this week; {len(added)} recurring non-sports/politics with activity added to the watchlist "
             f"({len(watch)} watched)."]
    for tk, title, cat, fr, st in sorted(added, key=lambda a: -a[4]["vol24h"])[:8]:
        lines.append(f"+ {tk} ({cat}, {fr}): {title[:50]} | {st['open_markets']} open, 24h vol {st['vol24h']:,.0f}, "
                     f"spread {st['med_spread_c']}c")
    for tk, w in graduated:
        t = w["test"]
        if "windows" in t:
            b = max(t["windows"].items(), key=lambda kv: kv[1].get("per_market_c") or -99)
            lines.append(f"{'PASS' if t['passes'] else 'fail'} {tk}: deep-fill edge {b[1].get('per_market_c')}c/market "
                         f"t {b[1].get('t')} ({b[0]}, {b[1].get('fills_per_mkt')} fills/market)")
        else:
            lines.append(f"tested {tk}: {t.get('result')}")
    if expired:
        lines.append(f"expired untested: {', '.join(expired[:10])}")
    notify("\n".join(lines))


if __name__ == "__main__":
    main()
