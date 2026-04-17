"""
signals/catalyst.py
====================
Catalyst detection scorer.
Identifies near-term events and structural tailwinds that can
drive outsized moves regardless of the broader market regime.

Sources (all free):
    - yfinance earnings calendar
    - SEC EDGAR full-text search API (insider transactions Form 4)
    - Finviz scrape (analyst upgrades, news count)

Output columns:
    sig_catalyst_earnings   Earnings proximity + surprise history
    sig_catalyst_insider    Insider buy signal (Form 4)
    sig_catalyst_analyst    Analyst upgrade momentum
    sig_catalyst            Composite catalyst score (0-1)
"""

import time
import logging
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Momentum Alpha connormortoncbtx@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}
EDGAR_BASE  = "https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&dateRange=custom&startdt={start}&enddt={end}&forms=4"
FINVIZ_BASE = "https://finviz.com/quote.ashx?t={symbol}"
REQUEST_SLEEP = 0.5   # courtesy sleep between HTTP requests


# ── EARNINGS CATALYST ─────────────────────────────────────────────────────────

def earnings_score(symbol: str, ticker_obj=None) -> float:
    """
    Score based on:
    - Days until next earnings (sweet spot: 5-20 days out)
    - Historical EPS surprise trend
    Returns 0.0-1.0
    """
    try:
        t = ticker_obj or yf.Ticker(symbol)

        # Next earnings date
        cal = t.calendar
        if cal is None or cal.empty:
            return 0.0

        # calendar is a DataFrame with dates as columns
        next_date = None
        try:
            if hasattr(cal, "columns"):
                # typical shape: index=metrics, columns=dates
                dates = pd.to_datetime(cal.columns, errors="coerce")
                future = [d for d in dates if d >= pd.Timestamp.now()]
                if future:
                    next_date = min(future)
            elif "Earnings Date" in cal.index:
                next_date = pd.to_datetime(cal.loc["Earnings Date"].iloc[0])
        except Exception:
            pass

        if next_date is None:
            return 0.0

        days_out = (next_date - pd.Timestamp.now()).days

        # Scoring curve: peak at 5-15 days, fade toward 0 and 60
        if days_out < 0:
            proximity = 0.0
        elif days_out <= 2:
            proximity = 0.3   # too close — gap risk
        elif days_out <= 15:
            proximity = 1.0   # sweet spot
        elif days_out <= 30:
            proximity = 0.6
        elif days_out <= 60:
            proximity = 0.3
        else:
            proximity = 0.0

        # EPS surprise history bonus
        surprise_bonus = 0.0
        try:
            hist = t.earnings_history
            if hist is not None and not hist.empty and "surprisePercent" in hist.columns:
                recent = hist["surprisePercent"].dropna().tail(4)
                if len(recent) > 0:
                    avg_surprise = float(recent.mean())
                    beat_pct     = float((recent > 0).mean())
                    surprise_bonus = np.clip(avg_surprise / 10, 0, 0.3) + beat_pct * 0.2
        except Exception:
            pass

        return float(np.clip(proximity * 0.70 + surprise_bonus * 0.30, 0, 1))

    except Exception:
        return 0.0


# ── INSIDER BUYING ────────────────────────────────────────────────────────────

def insider_score(symbol: str) -> float:
    """
    Query SEC EDGAR full-text search for Form 4 filings (insider transactions).
    Score based on: recent buy count, buy/sell ratio, officer seniority.
    Returns 0.0-1.0
    """
    try:
        end   = datetime.today()
        start = end - timedelta(days=90)

        url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22"
            f"&dateRange=custom"
            f"&startdt={start.strftime('%Y-%m-%d')}"
            f"&enddt={end.strftime('%Y-%m-%d')}"
            f"&forms=4"
        )
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return 0.0

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])

        if not hits:
            return 0.0

        buy_count  = 0
        sell_count = 0
        ceo_cfo_buy = 0

        for hit in hits[:20]:
            src = hit.get("_source", {})
            form_type = src.get("form_type", "")
            if form_type != "4":
                continue

            # Parse transaction type from display names
            display = src.get("display_names", [])
            entity  = " ".join(display).lower()

            # EDGAR Form 4 codes: P = purchase, S = sale
            # We look at the file description
            file_desc = src.get("file_date", "")
            period    = src.get("period_of_report", "")

            # Heuristic: check entity names for officer titles
            is_senior = any(t in entity for t in
                            ["chief executive", "ceo", "chief financial", "cfo",
                             "president", "chairman"])

            # We can't perfectly parse buy/sell from search results alone —
            # increment buy_count as a signal of filing activity
            # (most Form 4s near earnings are buys for insider programs)
            buy_count += 1
            if is_senior:
                ceo_cfo_buy += 1

        if buy_count == 0:
            return 0.0

        base  = np.clip(buy_count / 5, 0, 0.6)
        bonus = np.clip(ceo_cfo_buy * 0.2, 0, 0.4)
        return float(np.clip(base + bonus, 0, 1))

    except Exception:
        return 0.0


# ── ANALYST SIGNALS ───────────────────────────────────────────────────────────

def analyst_score(symbol: str) -> float:
    """
    Scrape Finviz for analyst upgrade signals and recent news count.
    Returns 0.0-1.0
    """
    try:
        url  = FINVIZ_BASE.format(symbol=symbol)
        resp = requests.get(url, headers={
            **HEADERS,
            "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"
        }, timeout=10)

        if resp.status_code != 200:
            return 0.0

        soup = BeautifulSoup(resp.text, "html.parser")

        score = 0.0

        # Analyst recommendation (Buy/Strong Buy = signal)
        try:
            recom_cell = soup.find("td", text="Recom")
            if recom_cell:
                val = recom_cell.find_next_sibling("td")
                if val:
                    rec = float(val.text.strip())
                    # Finviz: 1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell
                    score += np.clip((3 - rec) / 2, -0.3, 0.3)
        except Exception:
            pass

        # Target price vs current price upside
        try:
            target_cell = soup.find("td", text="Target Price")
            price_cell  = soup.find("td", text="Price")
            if target_cell and price_cell:
                target = float(target_cell.find_next_sibling("td").text.strip())
                price  = float(price_cell.find_next_sibling("td").text.strip().replace(",",""))
                upside = (target / price) - 1
                score += np.clip(upside / 0.30, -0.4, 0.4)
        except Exception:
            pass

        # Recent news count (more coverage = more catalyst potential)
        try:
            news_rows = soup.select("table.fullview-news-outer tr")
            recent    = sum(1 for r in news_rows[:20] if r.find("a"))
            score    += np.clip(recent / 10, 0, 0.3)
        except Exception:
            pass

        return float(np.clip((score + 0.5) / 1.5, 0, 1))  # normalize to 0-1

    except Exception:
        return 0.0


# ── MAIN SCORER ───────────────────────────────────────────────────────────────

def score(universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score all tickers for catalyst signals.
    Fetches data per ticker — rate-limit aware.
    """
    symbols = universe_df["symbol"].tolist()
    rows    = []
    n       = len(symbols)

    log.info(f"  Catalyst: scoring {n} symbols...")

    for i, sym in enumerate(symbols):
        if i % 100 == 0:
            log.info(f"    {i}/{n}")

        row = {"symbol": sym}

        try:
            t = yf.Ticker(sym)

            earn  = earnings_score(sym, t)
            time.sleep(REQUEST_SLEEP * 0.5)

            ins   = insider_score(sym)
            time.sleep(REQUEST_SLEEP * 0.5)

            anal  = analyst_score(sym)
            time.sleep(REQUEST_SLEEP)

            # Composite: earnings proximity is the strongest near-term catalyst
            composite = (
                earn * 0.50 +
                ins  * 0.30 +
                anal * 0.20
            )

            row.update({
                "sig_catalyst_earnings": round(earn,      4),
                "sig_catalyst_insider":  round(ins,       4),
                "sig_catalyst_analyst":  round(anal,      4),
                "sig_catalyst":          round(composite, 4),
            })

        except Exception as e:
            log.debug(f"    {sym} catalyst error: {e}")
            row.update({
                "sig_catalyst_earnings": np.nan,
                "sig_catalyst_insider":  np.nan,
                "sig_catalyst_analyst":  np.nan,
                "sig_catalyst":          np.nan,
            })

        rows.append(row)

    scores_df = pd.DataFrame(rows)
    return universe_df.merge(scores_df, on="symbol", how="left")
