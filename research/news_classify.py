"""
research/news_classify.py
=========================
Does an LLM reading the news around an earnings release add information
beyond the price reaction?

For each event in research/news_sample.csv (pre-registered random sample):
  1. Alpaca news (headline + summary) from the day before the filing through d1.
  2. Anonymise: company name tokens, ticker, years, month names, URLs removed,
     to reduce the model "remembering" what the stock did next.
  3. Claude labels the news with a fixed JSON schema (structured outputs).
Writes research_data/news_labels.parquet. Hard budget cap (BUDGET_USD).

Runs in GitHub Actions (research_news.yml): needs ALPACA_* and ANTHROPIC_API_KEY.
"""
import concurrent.futures as cf
import json
import logging
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger("news_classify")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

MODEL = os.environ.get("NEWS_MODEL", "claude-sonnet-5-5")
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}   # $/MTok in, out
BUDGET_USD = float(os.environ.get("BUDGET_USD", "40"))
MAX_ARTICLES = 25

SCHEMA = {
    "type": "object",
    "properties": {
        "news_relevant": {"type": "boolean"},
        "fundamental_signal": {"type": "integer", "enum": [-3, -2, -1, 0, 1, 2, 3]},
        "guidance": {"type": "string", "enum": ["raised", "maintained", "lowered", "none_mentioned"]},
        "driver_durability": {"type": "string", "enum": ["recurring", "one_time", "unclear"]},
        "reaction_assessment": {"type": "string",
                                "enum": ["underreacted", "appropriate", "overreacted", "unclear"]},
        "confidence": {"type": "number"},
        "key_driver": {"type": "string"},
    },
    "required": ["news_relevant", "fundamental_signal", "guidance", "driver_durability",
                 "reaction_assessment", "confidence", "key_driver"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are a buy-side equity analyst. You will read news published around one company's "
    "quarterly earnings release. Identifying details (company name, ticker, dates) have been "
    "removed on purpose. Do not try to identify the company or use any outside knowledge of "
    "what happened later; judge only from the text provided.\n\n"
    "Fields:\n"
    "- news_relevant: false if the articles say nothing about this company's results.\n"
    "- fundamental_signal: -3 (results/outlook clearly deteriorating) to +3 (clearly improving), "
    "judged on business fundamentals, not on the stock move.\n"
    "- guidance: direction of forward guidance if stated.\n"
    "- driver_durability: is the main driver likely to recur next quarter?\n"
    "- reaction_assessment: given the fundamentals, did the stated 2-day stock move under-react, "
    "react appropriately, or over-react?\n"
    "- confidence: 0 to 1.\n"
    "- key_driver: the main driver in 12 words or fewer."
)

CORP_WORDS = {"inc", "inc.", "corp", "corp.", "corporation", "company", "co", "co.", "ltd", "ltd.", "limited",
              "plc", "holdings", "holding", "group", "common", "stock", "class", "a", "b", "c", "ordinary",
              "shares", "share", "american", "depositary", "the", "and", "&", "n.v.", "s.a.", "lp", "l.p."}
# "may" only as a month when followed by a day number (otherwise it's the verb)
MONTHS = r"\b(january|february|march|april|june|july|august|september|october|november|december|" \
         r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?\b|\bmay(?=\s+\d)"


def anonymise(text: str, symbol: str, name: str) -> str:
    t = re.sub(r"https?://\S+", "", text)
    toks = [w for w in re.split(r"[\s,]+", name) if w and w.lower().strip(".") not in CORP_WORDS]
    for w in toks[:3]:
        if len(w) >= 3:
            t = re.sub(rf"\b{re.escape(w)}('s)?\b", "the Company", t, flags=re.I)
    t = re.sub(rf"(\$|\b(NYSE|NASDAQ|Nasdaq|AMEX)\s*:\s*)?\b{re.escape(symbol)}\b", "[TICKER]", t)
    t = re.sub(r"\b(19|20)\d{2}\b", "[YEAR]", t)
    t = re.sub(MONTHS, "[MONTH]", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def fetch_news(symbol, start, end):
    q = urllib.parse.urlencode({"symbols": symbol, "start": start, "end": end, "limit": 50,
                                "sort": "asc", "include_content": "false"})
    req = urllib.request.Request(f"https://data.alpaca.markets/v1beta1/news?{q}", headers={
        "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"], "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]})
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read()).get("news", [])
        except Exception as e:
            log.warning(f"  news {symbol}: {e}")
            time.sleep(3 * (i + 1))
    return None


class Budget:
    def __init__(self, cap):
        self.cap, self.spent, self.lock = cap, 0.0, threading.Lock()

    def add(self, usage):
        pin, pout = PRICES.get(MODEL, (4.0, 20.0))
        c = usage.input_tokens * pin / 1e6 + usage.output_tokens * pout / 1e6
        with self.lock:
            self.spent += c
            return self.spent

    halted = False

    def exceeded(self):
        return self.halted or self.spent >= self.cap


def label(client, budget, row):
    if budget.exceeded():
        return None
    start = (pd.Timestamp(row.pre_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00-05:00")
    end = pd.Timestamp(row.d1).strftime("%Y-%m-%dT23:59:59-05:00")
    news = fetch_news(row.symbol, start, end)
    base = {"symbol": row.symbol, "week": row.week, "split": row.split}
    if news is None:
        return {**base, "status": "news_error"}
    arts = [f"- {a.get('headline', '')}. {(a.get('summary') or '')[:400]}" for a in news[:MAX_ARTICLES]]
    if not arts:
        return {**base, "status": "no_news", "n_articles": 0}
    body = anonymise("\n".join(arts), row.symbol, row.name if isinstance(row.name, str) else "")
    move = f"{row.r2_abn * 100:+.1f}%"
    msg = (f"Stock move over the 2 trading days after the release, relative to the market: {move}\n\n"
           f"News ({len(arts)} articles):\n{body}")
    for attempt in range(4):
        try:
            r = client.messages.create(
                model=MODEL, max_tokens=2000, system=SYSTEM,
                thinking={"type": "adaptive"},
                output_config={"effort": "low",
                               "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": msg}],
            )
            spent = budget.add(r.usage)
            text = next(b.text for b in r.content if b.type == "text")
            out = json.loads(text)
            return {**base, "status": "ok", "n_articles": len(arts), **out,
                    "in_tok": r.usage.input_tokens, "out_tok": r.usage.output_tokens, "spent_usd": round(spent, 4)}
        except Exception as e:
            log.warning(f"  claude {row.symbol} {row.week}: {e}")
            if "credit" in str(e).lower() or "authentication" in str(e).lower():
                budget.halted = True           # stop everything; not retryable
                return {**base, "status": f"fatal: {str(e)[:120]}"}
            time.sleep(4 * (attempt + 1))
    return {**base, "status": "llm_error"}


def main():
    import anthropic
    data = Path(os.environ.get("RESEARCH_DATA", "research_data"))
    sample = pd.read_csv("research/news_sample.csv", parse_dates=["week", "pre_date", "d0", "d1"])
    lim = int(os.environ.get("LIMIT", "0") or 0)
    if lim:
        sample = sample.groupby("split").head(lim)
    prev = None
    if (data / "news_labels.parquet").exists():
        # Resume: keep finished rows, only redo events that never completed
        prev = pd.read_parquet(data / "news_labels.parquet")
        prev["week"] = pd.to_datetime(prev["week"])
        done = prev[prev.status.isin(["ok", "no_news"])]
        key = set(zip(done.symbol, done.week))
        sample = sample[[(s_, w) not in key for s_, w in zip(sample.symbol, sample.week)]]
        prev = done
        log.info(f"Resuming: {len(done):,} already labelled, {len(sample):,} to go")
    client = anthropic.Anthropic()
    budget = Budget(BUDGET_USD)
    log.info(f"Labelling {len(sample):,} events with {MODEL}, budget ${BUDGET_USD:.0f}")
    rows = []
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(label, client, budget, r) for r in sample.itertuples(index=False)]
        for i, fu in enumerate(cf.as_completed(futs)):
            res = fu.result()
            if res:
                rows.append(res)
            if i % 100 == 0:
                log.info(f"  {i:,}/{len(futs):,}  spent ${budget.spent:.2f}")
    out = pd.DataFrame(rows)
    if prev is not None:
        out = pd.concat([prev, out], ignore_index=True)
    out.to_parquet(data / "news_labels.parquet", index=False)
    st = out.status.value_counts().to_dict()
    log.info(f"Done. statuses={st} total spend ${budget.spent:.2f}")
    print(f"::notice::news labels statuses={st} spend=${budget.spent:.2f} model={MODEL}")
    (data / "README_news.md").write_text(f"news labels: {st}, spend ${budget.spent:.2f}, model {MODEL}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
