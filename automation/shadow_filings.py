"""
automation/shadow_filings.py
============================
Shadow #6 (no orders): live version of the LLM filing reader (research/llm_filings.py,
results in docs/2026-10_strategy_research.md). Runs twice each weekday.

  1. Read EDGAR's "latest 8-K filings" feed since the last run.
  2. Keep small caps: market cap $30M-$2B, price >= $1, >= $300k/day traded,
     and at least one item from the study's list (no earnings-only filings).
  3. Same anonymised text + same prompt + same model (Claude Haiku 4.5) as the study.
  4. Two hypothetical books, IWM-adjusted, equal weight per trade:
       LONG  contract wins with direction >= 1 (study: +2.6% / +5.0% at 5 / 20 days)
       SHORT direction <= -1 or 4+ red flags (study: -1.6% / -2.4%); tracked as the
             return a short would have earned before borrow costs
     Entry = the first regular-session OPEN after the judgement was made (never
     earlier than the moment we could have acted); exit = close 20 sessions later.
Files: data/shadow_filings_state.json (checkpoint + open trades),
       data/shadow_filings_judgements.jsonl (every judgement),
       data/shadow_filings_ledger.jsonl (closed trades).
Cost: about $0.003 per filing judged.
"""
import datetime as dt
import json
import logging
import os
import re
import sys
import time
import types
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import llm_filings as L  # noqa: E402  (fetch_one, SYSTEM, MODEL, sec_get)
from automation.system_logger import log_event, LogStatus  # noqa: E402

log = logging.getLogger("shadow_filings")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
ET = ZoneInfo("America/New_York")
DATA = ROOT / "data"
STATE = DATA / "shadow_filings_state.json"
JUDGE = DATA / "shadow_filings_judgements.jsonl"
LEDGER = DATA / "shadow_filings_ledger.jsonl"
ITEMS = {"1.01", "1.02", "1.03", "2.01", "2.03", "2.04", "2.05", "2.06", "3.01", "3.02",
         "4.01", "4.02", "5.01", "5.02", "7.01", "8.01"}
HOLD_SESSIONS = 20
MAX_JUDGE_PER_RUN = 250          # cost guard (~$0.75)
FEED = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb="
        "&owner=include&start={start}&count=100&output=atom")


def feed_entries(since: dt.datetime):
    """8-K entries accepted after `since` (aware ET datetime)."""
    out = []
    for start in range(0, 1000, 100):
        xml = L.sec_get(FEED.format(start=start)) or ""
        entries = re.findall(r"<entry>(.*?)</entry>", xml, flags=re.S)
        if not entries:
            break
        oldest = None
        for e in entries:
            title = re.search(r"<title>(.*?)</title>", e, flags=re.S)
            link = re.search(r'<link[^>]*href="([^"]+)"', e)
            upd = re.search(r"<updated>(.*?)</updated>", e)
            summ = re.search(r"<summary[^>]*>(.*?)</summary>", e, flags=re.S)
            if not (title and link and upd):
                continue
            t = dt.datetime.fromisoformat(upd.group(1).strip()).astimezone(ET)
            oldest = t if oldest is None else min(oldest, t)
            if t <= since or not title.group(1).strip().startswith("8-K"):
                continue
            m = re.search(r"/data/(\d+)/(\d{18})/", link.group(1))
            if not m:
                continue
            acc = f"{m.group(2)[:10]}-{m.group(2)[10:12]}-{m.group(2)[12:]}"
            items = set(re.findall(r"Item (\d\.\d\d)", summ.group(1) if summ else ""))
            out.append({"cik": int(m.group(1)), "accession": acc, "accepted": t.isoformat(),
                        "items": ",".join(sorted(items))})
        if oldest is not None and oldest <= since:
            break
    return list({o["accession"]: o for o in out}.values())


def ticker_map():
    raw = L.sec_get("https://www.sec.gov/files/company_tickers.json")
    m = {}
    for v in json.loads(raw).values():
        m.setdefault(int(v["cik_str"]), v["ticker"].upper().replace("-", "."))
    return m


def shares_out(cik):
    raw = L.sec_get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/dei/EntityCommonStockSharesOutstanding.json")
    if not raw:
        return None
    vals = [u for us in json.loads(raw).get("units", {}).values() for u in us]
    vals = [v for v in vals if v.get("val")]
    return float(sorted(vals, key=lambda v: v.get("filed", ""))[-1]["val"]) if vals else None


def revenue_ttm(cik):
    for tag in ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"):
        raw = L.sec_get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json")
        if raw:
            usd = json.loads(raw).get("units", {}).get("USD", [])
            fy = [u for u in usd if u.get("fp") == "FY" and u.get("form", "").startswith("10-K")]
            if fy:
                return float(sorted(fy, key=lambda v: v.get("filed", ""))[-1]["val"])
    return float("nan")


def daily_bars(symbols, start, end=None):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    c = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    frames = []
    syms = sorted(set(symbols))
    for i in range(0, len(syms), 200):
        try:
            req = StockBarsRequest(symbol_or_symbols=syms[i:i + 200], timeframe=TimeFrame.Day,
                                   start=dt.datetime.combine(start, dt.time()),
                                   end=dt.datetime.combine(end, dt.time()) if end else None,
                                   adjustment=Adjustment.ALL, feed=DataFeed.SIP)
            d = c.get_stock_bars(req).df
            if len(d):
                frames.append(d.reset_index()[["symbol", "timestamp", "open", "close", "volume"]])
        except Exception as e:
            log.warning(f"bars batch failed: {e}")
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", "open", "close", "volume"])
    b = pd.concat(frames)
    b["date"] = pd.to_datetime(b.timestamp).dt.tz_convert(ET).dt.date
    return b


def judge(client, row, mcap, rev):
    t = L.fetch_one(types.SimpleNamespace(accession=row["accession"], cik=row["cik"], symbol=row["symbol"]))
    if not t or len(t["text"]) < 200:
        return None
    revs = f"${rev / 1e6:,.0f}M" if np.isfinite(rev) else "unknown"
    user = (f"Company size: market cap ${mcap / 1e6:,.0f}M; trailing annual revenue {revs}; "
            f"SIC industry code unknown. 8-K items: {row['items']}.\n\nFILING TEXT:\n{t['text']}")
    msg = client.messages.create(model=L.MODEL, max_tokens=300, system=L.SYSTEM,
                                 messages=[{"role": "user", "content": user}])
    m = re.search(r"\{.*\}", msg.content[0].text, flags=re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except Exception:
        j = {}
    return {**j, "accepted_index": t["accepted_index"],
            "in_tokens": msg.usage.input_tokens, "out_tokens": msg.usage.output_tokens}


def main():
    now = dt.datetime.now(ET)
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "since": (now - dt.timedelta(hours=18)).isoformat(), "open": []}
    since = dt.datetime.fromisoformat(state["since"])

    # 1-2. new filings -> small-cap candidates
    entries = [e for e in feed_entries(since) if set(e["items"].split(",")) & ITEMS]
    tmap = ticker_map()
    for e in entries:
        e["symbol"] = tmap.get(e["cik"])
    entries = [e for e in entries if e["symbol"]]
    log.info(f"{len(entries)} new 8-Ks with study items and a ticker since {since:%Y-%m-%d %H:%M}")
    bars = daily_bars([e["symbol"] for e in entries] + ["IWM"], now.date() - dt.timedelta(days=40))
    stats = bars.groupby("symbol").apply(lambda g: pd.Series({
        "price": g.sort_values("date").close.iloc[-1],
        "adv": (g.close * g.volume).tail(20).mean()})) if len(bars) else pd.DataFrame()
    cands = []
    for e in entries:
        if e["symbol"] not in stats.index:
            continue
        s = stats.loc[e["symbol"]]
        if s.price < 1 or s.adv < 3e5:
            continue
        sh = shares_out(e["cik"])
        if not sh:
            continue
        mcap = sh * s.price
        if 30e6 <= mcap <= 2e9:
            cands.append((e, mcap))
    log.info(f"{len(cands)} small-cap candidates")

    # 3. judge
    import anthropic
    client = anthropic.Anthropic()
    n_long = n_short = 0
    with JUDGE.open("a") as jf:
        for e, mcap in cands[:MAX_JUDGE_PER_RUN]:
            try:
                j = judge(client, e, mcap, revenue_ttm(e["cik"]))
            except Exception as ex:
                log.warning(f"judge failed {e['accession']}: {ex}")
                continue
            if not j:
                continue
            decided = dt.datetime.now(ET)
            rec = {**e, "mcap": mcap, "decided": decided.isoformat(), **j}
            jf.write(json.dumps(rec, default=str) + "\n")
            d = pd.to_numeric(j.get("direction"), errors="coerce")
            flags = len(j.get("red_flags") or [])
            book = None
            if j.get("event_type") == "contract_win" and d >= 1:
                book, n_long = "long_contract_win", n_long + 1
            elif d <= -1 or flags >= 4:
                book, n_short = "short_bad_news", n_short + 1
            if book:
                state["open"].append({"book": book, "symbol": e["symbol"], "accession": e["accession"],
                                      "decided": decided.isoformat(), "event_type": j.get("event_type"),
                                      "summary": j.get("summary"), "entry_date": None, "entry": None})
    state["since"] = max([since] + [dt.datetime.fromisoformat(e["accepted"]) for e in entries]).isoformat()

    # 4. mark trades: fill entries at the first open after the decision; close after 20 sessions
    if state["open"]:
        first = min(dt.datetime.fromisoformat(o["decided"]).date() for o in state["open"])
        b = daily_bars([o["symbol"] for o in state["open"]] + ["IWM"], first - dt.timedelta(days=3))
        still = []
        with LEDGER.open("a") as lf:
            for o in state["open"]:
                g = b[b.symbol == o["symbol"]].sort_values("date")
                iwm = b[b.symbol == "IWM"].set_index("date")
                dec = dt.datetime.fromisoformat(o["decided"])
                sess = g[[dt.datetime.combine(d, dt.time(9, 30), ET) > dec for d in g.date]]
                if sess.empty:
                    still.append(o); continue
                d0 = sess.iloc[0]
                o["entry_date"], o["entry"] = str(d0.date), float(d0.open)
                if len(sess) <= HOLD_SESSIONS:
                    still.append(o); continue
                d1 = sess.iloc[HOLD_SESSIONS]
                r = float(d1.close / d0.open - 1)
                ri = float(iwm.loc[d1.date, "close"] / iwm.loc[d0.date, "open"] - 1) if (
                    d0.date in iwm.index and d1.date in iwm.index) else float("nan")
                sign = 1 if o["book"].startswith("long") else -1
                row = {**o, "exit_date": str(d1.date), "exit": float(d1.close), "ret": r, "iwm_ret": ri,
                       "book_ret_vs_iwm": sign * (r - ri)}
                lf.write(json.dumps(row) + "\n")
        state["open"] = still
    STATE.write_text(json.dumps(state, indent=1))
    msg = (f"Shadow filings: {len(cands)} judged-eligible, new long {n_long}, new short {n_short}, "
           f"open trades {len(state['open'])}")
    log.info(msg)
    log_event("shadow_filings", LogStatus.SUCCESS, msg)


if __name__ == "__main__":
    main()
