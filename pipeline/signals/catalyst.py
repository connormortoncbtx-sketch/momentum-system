"""
signals/catalyst.py
====================
Catalyst detection scorer.
Identifies near-term events and structural tailwinds that can
drive outsized moves regardless of the broader market regime.

Sources (all free):
    - yfinance earnings calendar + earnings_dates
    - yfinance insider_transactions (replaces broken EDGAR scrape)
    - Finviz scrape (analyst upgrades, news count)

Output columns:
    sig_catalyst_earnings   Earnings proximity + surprise history
    sig_catalyst_insider    Insider buy signal (Form 4 derived)
    sig_catalyst_analyst    Analyst upgrade momentum
    sig_catalyst            Composite catalyst score (0-1)

──────────────────────────────────────────────────────────────────────────────
2026-04-25 PATCH NOTES — three independent bugs found via signal distribution
analysis (sig_catalyst_earnings and sig_catalyst_insider were 0.0 for 100% of
2,980 universe rows; sig_catalyst_analyst partially working).

1. earnings_score: yfinance>=0.2.30 returns Ticker.calendar as a DICT, not a
   DataFrame. The previous `cal.empty` check raised AttributeError on every
   call; the outer try/except swallowed it and returned 0.0 universally. Now
   handles both dict (modern) and DataFrame (legacy) formats. Also switched
   from `t.earnings_history` to `t.earnings_dates` for the surprise bonus
   since `earnings_history` is unreliable across yfinance versions.

2. insider_score: previously queried SEC EDGAR full-text search for the
   ticker symbol as literal text inside Form 4 filings. Form 4s are indexed
   by CIK/company-name and do not contain ticker symbols in their searchable
   body, so every query returned zero hits. Replaced entirely with
   yfinance's structured Ticker.insider_transactions, which provides
   buy/sell classification directly. Also kills one HTTP request per ticker.

3. analyst_score: BeautifulSoup's `text=` kwarg was deprecated (4.4) and
   removed (4.13+). Three call sites silently failed to match in modern bs4,
   leaving only the news-count component active. Renamed to `string=` which
   has been canonical since 2015.
──────────────────────────────────────────────────────────────────────────────
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
FINVIZ_BASE = "https://finviz.com/quote.ashx?t={symbol}"
REQUEST_SLEEP = 0.5   # courtesy sleep between HTTP requests


# ── EARNINGS CATALYST ─────────────────────────────────────────────────────────

def earnings_score(symbol: str, ticker_obj=None) -> float:
    """
    Score based on:
    - Days until next earnings (sweet spot: 5-15 days out)
    - Historical EPS surprise trend
    Returns 0.0-1.0
    """
    try:
        t = ticker_obj or yf.Ticker(symbol)

        # Modern yfinance returns calendar as a DICT, older as DataFrame.
        # Treat falsy (None, empty dict, empty df) as no data.
        cal = t.calendar
        if not cal:
            return 0.0

        next_date = None

        # Modern yfinance (>=0.2.30): dict shape
        # { 'Earnings Date': [Timestamp, Timestamp, ...], 'Earnings Average': ..., ... }
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date") or []
            if not isinstance(dates, list):
                dates = [dates]
            future = []
            for d in dates:
                try:
                    ts = pd.Timestamp(d)
                    if ts >= pd.Timestamp.now():
                        future.append(ts)
                except Exception:
                    continue
            if future:
                next_date = min(future)

        # Legacy yfinance: DataFrame with dates as columns
        elif hasattr(cal, "columns"):
            try:
                dates = pd.to_datetime(cal.columns, errors="coerce")
                future = [d for d in dates if pd.notna(d) and d >= pd.Timestamp.now()]
                if future:
                    next_date = min(future)
            except Exception:
                pass
            if next_date is None and hasattr(cal, "index") and "Earnings Date" in cal.index:
                try:
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
        # Switched from t.earnings_history to t.earnings_dates -- the former
        # is unreliable across yfinance versions; the latter is the modern
        # canonical attribute and has 'Surprise(%)' column.
        surprise_bonus = 0.0
        try:
            hist = getattr(t, "earnings_dates", None)
            if hist is None:
                hist = getattr(t, "earnings_history", None)

            if hist is not None and hasattr(hist, "empty") and not hist.empty:
                # Find the surprise column -- name varies across versions
                surprise_col = None
                for candidate in ("Surprise(%)", "surprisePercent", "Earnings Surprise %"):
                    if candidate in hist.columns:
                        surprise_col = candidate
                        break

                if surprise_col is not None:
                    # Past quarters only -- earnings_dates has future rows too,
                    # which have NaN surprise. dropna handles that automatically.
                    recent = pd.to_numeric(hist[surprise_col], errors="coerce").dropna().tail(4)
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

def insider_score(symbol: str, ticker_obj=None) -> float:
    """
    Score insider buying activity using yfinance's structured insider data.

    Replaces the previous SEC EDGAR full-text search approach, which was
    fundamentally broken: Form 4 filings are indexed by CIK/company name and
    do NOT contain ticker symbols in their searchable text, so queries like
    `q="AAPL"&forms=4` returned zero hits for virtually every ticker.

    yfinance.Ticker.insider_transactions provides parsed Form 4 data with
    explicit buy/sell classification, which is exactly what this signal
    needs and what the EDGAR approach was failing to extract.

    Returns 0.0-1.0
    """
    try:
        t = ticker_obj or yf.Ticker(symbol)

        tx = getattr(t, "insider_transactions", None)
        if tx is None or not hasattr(tx, "empty") or tx.empty:
            return 0.0

        # Filter to last 90 days. Date column name has varied across yfinance
        # versions — try the common candidates.
        date_col = None
        for cand in ("Start Date", "Date", "Transaction Date"):
            if cand in tx.columns:
                date_col = cand
                break

        if date_col is not None:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
            try:
                dates = pd.to_datetime(tx[date_col], errors="coerce")
                tx = tx[dates >= cutoff]
            except Exception:
                pass  # if date parse fails, just use everything

        if tx.empty:
            return 0.0

        # Buy/sell classification. Common column names across yfinance versions.
        txn_col = None
        for cand in ("Transaction", "Transaction Type", "Type"):
            if cand in tx.columns:
                txn_col = cand
                break

        if txn_col is None:
            return 0.0

        txn_strs = tx[txn_col].astype(str).str.lower()
        # yfinance commonly emits "Purchase" / "Sale" / "Sale (Multiple)" /
        # "Purchase at price" / "Stock Gift" etc. Match on substrings.
        buys  = tx[txn_strs.str.contains("purchase", na=False)]
        sells = tx[txn_strs.str.contains("sale",     na=False)]

        n_buys, n_sells = len(buys), len(sells)
        if n_buys == 0:
            return 0.0

        # Net seller -- no signal even if there's some buying activity
        total = n_buys + n_sells
        net_ratio = (n_buys - n_sells) / max(total, 1)         # range -1..1
        if net_ratio <= 0:
            return 0.0

        # CEO/CFO/President buys carry more weight than mid-level officers.
        # Position column names also vary -- try the common ones.
        ceo_cfo_buys = 0
        position_col = None
        for cand in ("Position", "Insider Title", "Title", "Filer Title"):
            if cand in buys.columns:
                position_col = cand
                break

        if position_col is not None:
            positions = buys[position_col].astype(str).str.lower()
            senior_mask = positions.str.contains(
                "ceo|chief executive|cfo|chief financial|president|chairman|director",
                na=False, regex=True,
            )
            ceo_cfo_buys = int(senior_mask.sum())

        base  = np.clip(n_buys / 5,           0, 0.6)
        bonus = np.clip(net_ratio * 0.3,      0, 0.3) + np.clip(ceo_cfo_buys * 0.05, 0, 0.1)
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

        # NOTE: BeautifulSoup's `text=` kwarg was deprecated (4.4) and removed
        # entirely in 4.13+. Renamed to `string=` here -- has been the canonical
        # kwarg name since 2015. Without this fix, two of three analyst sub-
        # components silently no-op'd and the score reduced to news-count only.

        # Analyst recommendation (Buy/Strong Buy = signal)
        try:
            recom_cell = soup.find("td", string="Recom")
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
            target_cell = soup.find("td", string="Target Price")
            price_cell  = soup.find("td", string="Price")
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
            # Single Ticker object reused across earnings + insider to avoid
            # constructing two yfinance handles per symbol.
            t = yf.Ticker(sym)

            earn  = earnings_score(sym, t)
            time.sleep(REQUEST_SLEEP * 0.5)

            ins   = insider_score(sym, t)
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
