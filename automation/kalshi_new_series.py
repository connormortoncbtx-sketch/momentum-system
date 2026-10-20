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
    hits = new[t.str.contains(KEYWORDS) & ~t.str.contains(NOT_KEYWORDS)]
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


def main():
    S = all_series()
    seen = set(json.loads(SEEN.read_text())) if SEEN.exists() else None
    keyword_check(S, seen)
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
