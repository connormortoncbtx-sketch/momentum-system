"""
signals/sentiment.py
=====================
Sentiment + narrative scorer.
Aggregates news velocity, headline tone, and social signal proxies
from free sources.

Sources:
    - Yahoo Finance RSS feed per ticker
    - Finviz news scrape (headline count + recency)
    - yfinance recommendation trend (analyst sentiment shift)

Output columns:
    sig_sentiment_news      News volume + tone
    sig_sentiment_analyst   Analyst recommendation trend (improving/worsening)
    sig_sentiment           Composite (0-1)
"""

import re
import time
import logging
import numpy as np
import pandas as pd
import feedparser
import requests
import yfinance as yf
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)",
}

# Positive / negative keyword sets for headline scoring
BULLISH_WORDS = {
    "beat", "beats", "record", "surge", "soars", "upgrade", "upgraded",
    "raises", "raised", "buyout", "acquisition", "partnership", "wins",
    "awarded", "breakthrough", "strong", "exceeds", "outperforms",
    "growth", "profit", "approval", "approved", "launch", "expands",
}
BEARISH_WORDS = {
    "miss", "misses", "missed", "downgrade", "downgraded", "lowers", "lowered",
    "loss", "losses", "cuts", "cut", "weak", "disappoints", "warning",
    "recall", "lawsuit", "investigation", "fraud", "decline", "falling",
    "bankruptcy", "default", "layoffs", "restructuring",
}


# ── NEWS SENTIMENT ────────────────────────────────────────────────────────────

def score_headline(title: str) -> float:
    """Simple keyword sentiment score for a single headline. -1 to +1."""
    words = set(re.sub(r"[^a-z\s]", "", title.lower()).split())
    pos   = len(words & BULLISH_WORDS)
    neg   = len(words & BEARISH_WORDS)
    if pos + neg == 0:
        return 0.0
    return float((pos - neg) / (pos + neg))


def fetch_yahoo_rss(symbol: str, max_items: int = 20) -> list[dict]:
    """Pull Yahoo Finance RSS for a ticker. Returns list of {title, published}."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    try:
        feed  = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                "title":     entry.get("title", ""),
                "published": entry.get("published", ""),
            })
        return items
    except Exception:
        return []


def news_score(symbol: str) -> tuple[float, int]:
    """
    Score news sentiment for a ticker.
    Returns (score 0-1, article_count).
    """
    items = fetch_yahoo_rss(symbol)

    if not items:
        return 0.5, 0   # neutral if no news

    # Recency weighting: more recent = higher weight
    now    = datetime.utcnow()
    scores = []
    for item in items:
        headline_score = score_headline(item["title"])

        # Parse published date for recency weight
        try:
            pub   = datetime(*item["published"][:6]) if item["published"] else now
            age_d = max(0, (now - pub).days)
            weight = max(0.1, 1.0 - (age_d / 14))  # decay over 14 days
        except Exception:
            weight = 0.5

        scores.append(headline_score * weight)

    if not scores:
        return 0.5, 0

    raw_sentiment = float(np.mean(scores))
    # Rescale from [-1,1] to [0,1], with article count bonus
    count_bonus   = np.clip(len(items) / 20, 0, 0.15) if raw_sentiment > 0 else 0
    normalized    = (raw_sentiment + 1) / 2 + count_bonus

    return float(np.clip(normalized, 0, 1)), len(items)


# ── ANALYST RECOMMENDATION TREND ─────────────────────────────────────────────

def analyst_trend_score(symbol: str, ticker_obj=None) -> float:
    """
    Measures whether analyst recommendations are improving or worsening.
    Uses yfinance recommendation trend (strongBuy, buy, hold, sell, strongSell counts).
    Returns 0-1 where 1 = strong improvement trend.
    """
    try:
        t    = ticker_obj or yf.Ticker(symbol)
        recs = t.recommendations_summary

        if recs is None or recs.empty:
            return 0.5

        # recommendations_summary has columns: period, strongBuy, buy, hold, sell, strongSell
        # periods: 0m (current), -1m, -2m, -3m
        if "period" in recs.columns:
            recs = recs.set_index("period")

        def bull_ratio(row):
            total = sum([
                row.get("strongBuy", 0), row.get("buy", 0),
                row.get("hold", 0), row.get("sell", 0), row.get("strongSell", 0)
            ])
            if total == 0:
                return 0.5
            bull = row.get("strongBuy", 0) * 1.0 + row.get("buy", 0) * 0.7
            bear = row.get("strongSell", 0) * 1.0 + row.get("sell", 0) * 0.7
            return float((bull - bear + total) / (2 * total))

        ratios = {}
        for period in ["0m", "-1m", "-2m", "-3m"]:
            if period in recs.index:
                ratios[period] = bull_ratio(recs.loc[period])

        if not ratios:
            return 0.5

        current = ratios.get("0m", 0.5)

        # Trend: is current better than 3 months ago?
        oldest  = ratios.get("-3m", ratios.get("-2m", ratios.get("-1m", current)))
        trend   = current - oldest

        return float(np.clip(current * 0.60 + (trend + 0.5) * 0.40, 0, 1))

    except Exception:
        return 0.5


# ── SHORT INTEREST PROXY ──────────────────────────────────────────────────────

def short_interest_score(info: dict) -> float:
    """
    Short interest as % of float — high short interest can be a squeeze setup
    when combined with positive momentum.
    Returns 0-1 where higher = more squeeze potential.
    """
    try:
        short_pct = info.get("shortPercentOfFloat")
        if short_pct is None:
            return 0.3  # neutral

        # >20% short float = meaningful squeeze potential
        # But extreme (>50%) can mean serious problems — cap benefit
        if short_pct > 0.50:
            return 0.5   # too risky, discount
        elif short_pct > 0.25:
            return 0.8
        elif short_pct > 0.10:
            return 0.6
        else:
            return 0.3   # low short interest = less squeeze but also less crowded

    except Exception:
        return 0.3


# ── MAIN SCORER ───────────────────────────────────────────────────────────────

def score(universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score all tickers for sentiment signals.
    """
    symbols = universe_df["symbol"].tolist()
    rows    = []
    n       = len(symbols)

    log.info(f"  Sentiment: scoring {n} symbols...")

    for i, sym in enumerate(symbols):
        if i % 100 == 0:
            log.info(f"    {i}/{n}")

        row = {"symbol": sym}

        try:
            t    = yf.Ticker(sym)
            info = t.info or {}

            news_s, news_count = news_score(sym)
            time.sleep(0.2)

            analyst_s = analyst_trend_score(sym, t)
            short_s   = short_interest_score(info)

            composite = (
                news_s    * 0.50 +
                analyst_s * 0.35 +
                short_s   * 0.15
            )

            row.update({
                "sig_sentiment_news":     round(news_s,    4),
                "sig_sentiment_analyst":  round(analyst_s, 4),
                "sig_sentiment_short":    round(short_s,   4),
                "sig_sentiment_articles": news_count,
                "sig_sentiment":          round(composite, 4),
            })

        except Exception as e:
            log.debug(f"    {sym} sentiment error: {e}")
            row.update({
                "sig_sentiment_news":     np.nan,
                "sig_sentiment_analyst":  np.nan,
                "sig_sentiment_short":    np.nan,
                "sig_sentiment_articles": 0,
                "sig_sentiment":          np.nan,
            })

        rows.append(row)

    scores_df = pd.DataFrame(rows)
    return universe_df.merge(scores_df, on="symbol", how="left")
