"""
research/llm_filings.py -- "read every small-cap filing, scaled to the company" study.

Runs in GitHub Actions (research_llm_filings.yml). Three checkpointed stages;
a re-run skips any stage whose output already exists.

  A. text     for each filing in research/samples/llm_8k_sample.parquet, read the
              EDGAR index page (authoritative 'Accepted' time, Eastern) and the
              8-K body + EX-99 exhibits; strip HTML; ANONYMISE (company name and
              ticker replaced) so the model cannot recall what the stock did;
              keep the first ~9,000 characters.        -> llm/texts.parquet
  B. llm      Claude Haiku 4.5 via the Message Batches API reads each filing with
              the company's market cap and trailing revenue and returns a JSON
              judgement (type, $ value, direction, materiality, novelty, red
              flags).                                    -> llm/judgements.parquet
  C. returns  Alpaca daily bars: entry at the FIRST OPEN after acceptance, exits
              at that day's close and 1/5/20 sessions later; IWM-adjusted.
                                                         -> llm/events.parquet
Holdout = filings after the model's training cutoff (Apr 2025 on), so the model
cannot have seen what happened next. Dev (2023 - Mar 2025) is for design only.
"""
import concurrent.futures as cf
import datetime as dt
import html
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("llm_filings")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
UA = os.environ.get("SEC_USER_AGENT") or "MomentumAlpha research momentum-bot@users.noreply.github.com"
# Full-coverage backfill: SAMPLE=research/samples/llm_8k_full.parquet LLM_DIR=llm_full PART=i NPARTS=n
# (each part checkpoints separately under <LLM_DIR>/part<i>)
PART, NPARTS = int(os.environ.get("PART") or 0), int(os.environ.get("NPARTS") or 1)
OUT = Path(os.environ.get("RESEARCH_DATA", "research_data")) / (os.environ.get("LLM_DIR") or "llm")
if NPARTS > 1:
    OUT = OUT / f"part{PART}"
SAMPLE = Path(os.environ.get("SAMPLE") or "research/samples/llm_8k_sample.parquet")
MODEL = os.environ.get("LLM_MODEL") or "claude-haiku-4-5-20251001"
MAX_CHARS = 9000


class Limiter:
    def __init__(self, per_sec):
        self.gap, self.lock, self.t = 1.0 / per_sec, threading.Lock(), 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.t:
                time.sleep(self.t - now)
            self.t = max(now, self.t) + self.gap


SEC = Limiter(8)


def sec_get(url, tries=4):
    for i in range(tries):
        SEC.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(4 * (i + 1))
        except Exception:
            time.sleep(4 * (i + 1))
    return None


TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")


def to_text(h):
    h = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", h)
    h = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", h)
    return WS.sub(" ", html.unescape(TAG.sub(" ", h))).strip()


def anonymise(text, name, symbol):
    out = text
    if name:
        core = re.sub(r"(?i)[,.]?\s+(inc|corp|corporation|ltd|limited|plc|co|company|holdings?|group|n\.?v|s\.?a|llc|l\.?p)\.?$", "", name.strip())
        for pat in {name.strip(), core}:
            if len(pat) >= 3:
                out = re.sub(re.escape(pat), "the Company", out, flags=re.I)
    if symbol and len(symbol) >= 2:
        out = re.sub(rf"\b{re.escape(symbol)}\b", "TICKER", out)
    return out


def fetch_one(row):
    acc = row.accession
    base = f"https://www.sec.gov/Archives/edgar/data/{int(row.cik)}/{acc.replace('-', '')}"
    idx = sec_get(f"{base}/{acc}-index.htm")
    if not idx:
        return None
    m = re.search(r'Accepted</div>\s*<div class="info">([^<]+)<', idx)
    accepted = m.group(1).strip() if m else None
    nm = re.search(r'<span class="companyName">([^<(]+)', idx)
    name = nm.group(1).strip() if nm else ""
    docs = re.findall(r'<td[^>]*>([^<]*)</td>\s*<td[^>]*><a href="([^"]+)">[^<]*</a>.*?</td>\s*<td[^>]*>([^<]*)</td>',
                      idx, flags=re.S)
    picks = []
    for desc, href, typ in docs:
        t = typ.strip().upper()
        if (t.startswith("8-K") or t.startswith("EX-99")) and href.lower().endswith((".htm", ".html", ".txt")):
            picks.append((0 if t.startswith("8-K") else 1, href))
    texts = []
    for _, href in sorted(picks)[:3]:
        url = href if href.startswith("http") else "https://www.sec.gov" + href.replace("/ix?doc=", "")
        h = sec_get(url)
        if h:
            texts.append(to_text(h))
    text = anonymise("\n\n".join(texts), name, row.symbol)[:MAX_CHARS]
    return {"accession": acc, "accepted_index": accepted, "n_docs": len(texts), "text": text}


def stage_text(sample):
    f = OUT / "texts.parquet"
    if f.exists():
        log.info("texts exist -- skip stage A")
        return pd.read_parquet(f)
    rows, done = [], 0
    with cf.ThreadPoolExecutor(6) as ex:
        for r in ex.map(fetch_one, sample.itertuples(index=False)):
            done += 1
            if r:
                rows.append(r)
            if done % 1000 == 0:
                log.info(f"  text {done:,}/{len(sample):,}")
    t = pd.DataFrame(rows)
    t.to_parquet(f, index=False, compression="zstd")
    log.info(f"texts: {len(t):,} filings, median {t.text.str.len().median():.0f} chars")
    return t


SYSTEM = """You are an equity analyst screening SEC 8-K filings of small US-listed companies.
The company's name and ticker are hidden. Judge ONLY from the filing text and the size context given.
Return ONLY a JSON object, no prose:
{"event_type": one of ["contract_win","contract_loss","partnership","acquisition","divestiture","debt_financing",
  "equity_financing","dilution_other","guidance_raise","guidance_cut","prelim_results_good","prelim_results_bad",
  "exec_departure","exec_appointment","auditor_change","restatement","listing_deficiency","going_concern",
  "regulatory_approval","regulatory_setback","clinical_positive","clinical_negative","litigation",
  "restructuring","buyback_dividend","strategic_review","routine","other"],
 "dollar_value_usd": number or null (headline $ value of the deal/contract/financing, if stated),
 "direction": integer -2..2 (likely effect on the share price over the next week; 0 = none),
 "materiality": number 0..1 (importance relative to THIS company's size),
 "novelty": number 0..1 (genuinely new information vs routine/expected/already announced),
 "red_flags": list of short strings (e.g. "auditor resigned", "going concern", "toxic convertible", "CFO left abruptly"),
 "summary": string (max 20 words)}"""


def stage_llm(sample, texts):
    f = OUT / "judgements.parquet"
    if f.exists():
        log.info("judgements exist -- skip stage B")
        return pd.read_parquet(f)
    import anthropic
    client = anthropic.Anthropic()
    ctx = sample.set_index("accession")
    reqs = []
    for r in texts.itertuples(index=False):
        if len(r.text) < 200:
            continue
        c = ctx.loc[r.accession]
        rev = f"${c.revenue_ttm / 1e6:,.0f}M" if np.isfinite(c.revenue_ttm) else "unknown"
        user = (f"Company size: market cap ${c.mcap / 1e6:,.0f}M; trailing annual revenue {rev}; "
                f"SIC industry code {c.sic or 'unknown'}. 8-K items: {c['items']}.\n\nFILING TEXT:\n{r.text}")
        reqs.append({"custom_id": r.accession.replace("-", "_"),
                     "params": {"model": MODEL, "max_tokens": 300, "system": SYSTEM,
                                "messages": [{"role": "user", "content": user}]}})
    log.info(f"LLM requests: {len(reqs):,}")
    batch_ids = []
    for i in range(0, len(reqs), 8000):                 # stay well under batch size limits
        b = client.messages.batches.create(requests=reqs[i:i + 8000])
        batch_ids.append(b.id)
        log.info(f"  batch {b.id}: {len(reqs[i:i + 8000]):,} requests")
    (OUT / "batches.json").write_text(json.dumps(batch_ids))
    t0 = time.time()
    while True:
        st = [client.messages.batches.retrieve(b) for b in batch_ids]
        if all(s.processing_status == "ended" for s in st):
            break
        if time.time() - t0 > 4.5 * 3600:
            raise RuntimeError("batches not finished in 4.5h -- re-run later to collect")
        log.info("  waiting: " + ", ".join(f"{s.request_counts.processing} left" for s in st))
        time.sleep(90)
    rows = []
    for b in batch_ids:
        for res in client.messages.batches.results(b):
            acc = res.custom_id.replace("_", "-")
            if res.result.type != "succeeded":
                rows.append({"accession": acc, "ok": False})
                continue
            txt = res.result.message.content[0].text
            m = re.search(r"\{.*\}", txt, flags=re.S)
            try:
                j = json.loads(m.group(0)) if m else {}
            except Exception:
                j = {}
            rows.append({"accession": acc, "ok": bool(j), "event_type": j.get("event_type"),
                         "dollar_value_usd": pd.to_numeric(j.get("dollar_value_usd"), errors="coerce"),
                         "direction": pd.to_numeric(j.get("direction"), errors="coerce"),
                         "materiality": pd.to_numeric(j.get("materiality"), errors="coerce"),
                         "novelty": pd.to_numeric(j.get("novelty"), errors="coerce"),
                         "red_flags": "|".join(map(str, j.get("red_flags") or [])),
                         "summary": j.get("summary"),
                         "in_tokens": res.result.message.usage.input_tokens,
                         "out_tokens": res.result.message.usage.output_tokens})
    jd = pd.DataFrame(rows)
    jd.to_parquet(f, index=False)
    log.info(f"judgements: {jd.ok.sum():,} ok of {len(jd):,}; tokens in {jd.in_tokens.sum():,} out {jd.out_tokens.sum():,}")
    return jd


def stage_returns(sample, texts):
    f = OUT / "events.parquet"
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    syms = sorted(set(sample.symbol) | {"IWM", "SPY"})
    start = sample.accepted_et.min() - pd.Timedelta(days=10)
    frames = []
    for i in range(0, len(syms), 200):
        try:
            req = StockBarsRequest(symbol_or_symbols=syms[i:i + 200], timeframe=TimeFrame.Day,
                                   start=start.to_pydatetime(), adjustment=Adjustment.ALL, feed=DataFeed.SIP)
            d = client.get_stock_bars(req).df
            if len(d):
                frames.append(d.reset_index()[["symbol", "timestamp", "open", "close", "volume"]])
        except Exception as e:
            log.warning(f"  bars batch {i // 200}: {e}")
    bars = pd.concat(frames)
    bars["date"] = pd.to_datetime(bars.timestamp).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    O = bars.pivot_table(index="date", columns="symbol", values="open")
    C = bars.pivot_table(index="date", columns="symbol", values="close")
    days = C.index

    ev = sample.merge(texts[["accession", "accepted_index", "n_docs"]], on="accession", how="left")
    acc_t = pd.to_datetime(ev.accepted_index, errors="coerce").fillna(ev.accepted_et)
    ev["accepted_true"] = acc_t
    # first session whose 09:30 open is after acceptance
    open_t = days + pd.Timedelta(hours=9, minutes=30)
    pos = np.searchsorted(open_t.values, acc_t.values.astype("datetime64[ns]"), side="right")
    out = {k: np.full(len(ev), np.nan) for k in ("gap", "r_oc0", "r_1", "r_5", "r_20", "ab_oc0", "ab_1", "ab_5", "ab_20")}
    for n, (sym, p) in enumerate(zip(ev.symbol, pos)):
        if sym not in C.columns or p >= len(days) or p < 1:
            continue
        o0 = O[sym].iloc[p]
        pc = C[sym].iloc[p - 1]
        if not (np.isfinite(o0) and o0 > 0):
            continue
        out["gap"][n] = o0 / pc - 1 if np.isfinite(pc) and pc > 0 else np.nan
        bo = O["IWM"].iloc[p]
        for key, h in (("oc0", 0), ("1", 1), ("5", 5), ("20", 20)):
            if p + h < len(days):
                c1, bc = C[sym].iloc[p + h], C["IWM"].iloc[p + h]
                r = c1 / o0 - 1
                out[f"r_{key}"][n] = r
                out[f"ab_{key}"][n] = r - (bc / bo - 1)
    for k, v in out.items():
        ev[k] = v
    ev.to_parquet(f, index=False)
    log.info(f"events with returns: {np.isfinite(ev.r_1).sum():,} of {len(ev):,}")
    return ev


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sample = pd.read_parquet(SAMPLE).drop_duplicates("accession").reset_index(drop=True)
    if NPARTS > 1:
        sample = sample[sample.index % NPARTS == PART]
    only = (os.environ.get("ONLY") or "text,llm,returns").split(",")
    texts = stage_text(sample) if "text" in only else pd.read_parquet(OUT / "texts.parquet")
    notes = [f"texts {len(texts):,}"]
    if "llm" in only:
        jd = stage_llm(sample, texts)
        notes.append(f"judgements {int(jd.ok.sum()):,} ok; tokens in {int(jd.in_tokens.sum()):,} out {int(jd.out_tokens.sum()):,}")
    if "returns" in only:
        ev = stage_returns(sample, texts)
        notes.append(f"events with returns {int(np.isfinite(ev.r_1).sum()):,}")
    print("::notice::llm filings: " + "; ".join(notes))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
