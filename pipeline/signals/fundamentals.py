"""
signals/fundamentals.py
========================
Fundamentals quality scorer.
Identifies companies with accelerating growth, healthy balance sheets,
and improving profitability — the underlying business momentum
that sustains price momentum long-term.

Sources: yfinance (free)

Output columns:
    sig_fund_growth         Revenue + earnings growth trend
    sig_fund_quality        Balance sheet health
    sig_fund_profitability  Margin trend + ROE
    sig_fund_value          Relative valuation (not cheap-hunting, penalizes extremes)
    sig_fundamentals        Composite (0-1)
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def safe_get(info: dict, key: str, default=None):
    v = info.get(key, default)
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else v


def acceleration(series: list[float]) -> float:
    """
    Measures whether growth is accelerating.
    Returns +1 if each value is larger than previous, -1 if decelerating.
    """
    if len(series) < 2:
        return 0.0
    diffs  = [series[i] - series[i-1] for i in range(1, len(series))]
    pos    = sum(1 for d in diffs if d > 0)
    neg    = sum(1 for d in diffs if d < 0)
    return float((pos - neg) / len(diffs))


# ── GROWTH SCORER ─────────────────────────────────────────────────────────────

def growth_score(info: dict, financials: pd.DataFrame) -> float:
    """
    Revenue growth (YoY + trend) + EPS growth.
    Rewards acceleration, not just magnitude.
    """
    score = 0.0

    # YoY revenue growth from info
    rev_growth = safe_get(info, "revenueGrowth")
    if rev_growth is not None:
        score += np.clip(rev_growth / 0.20, -1, 1) * 0.25

    # EPS growth (TTM)
    eps_growth = safe_get(info, "earningsGrowth")
    if eps_growth is not None:
        score += np.clip(eps_growth / 0.20, -1, 1) * 0.25

    # Revenue acceleration from quarterly financials
    if financials is not None and not financials.empty:
        try:
            if "Total Revenue" in financials.index:
                rev_series = financials.loc["Total Revenue"].dropna().tolist()
                if len(rev_series) >= 3:
                    # Most recent quarters first — reverse for chronological
                    rev_series = list(reversed(rev_series))
                    accel = acceleration(rev_series[-4:])
                    score += accel * 0.25
        except Exception:
            pass

    # Forward EPS estimate revision (analysts raising = signal)
    eps_fwd  = safe_get(info, "forwardEps")
    eps_trail = safe_get(info, "trailingEps")
    if eps_fwd and eps_trail and eps_trail > 0:
        revision = (eps_fwd / eps_trail) - 1
        score   += np.clip(revision / 0.15, -0.5, 0.5) * 0.25

    return float(np.clip((score + 1) / 2, 0, 1))


# ── QUALITY SCORER ────────────────────────────────────────────────────────────

def quality_score(info: dict) -> float:
    """
    Balance sheet health: debt levels, current ratio, cash position.
    """
    score = 0.0

    # Debt/equity (lower is better for momentum; penalize overleveraged)
    de = safe_get(info, "debtToEquity")
    if de is not None:
        if de < 0:
            score += -0.3   # negative equity is a red flag
        elif de < 0.3:
            score += 0.4
        elif de < 0.8:
            score += 0.2
        elif de < 1.5:
            score += 0.0
        elif de < 3.0:
            score += -0.2
        else:
            score += -0.4

    # Current ratio (liquidity)
    cr = safe_get(info, "currentRatio")
    if cr is not None:
        if cr >= 2.0:
            score += 0.3
        elif cr >= 1.5:
            score += 0.2
        elif cr >= 1.0:
            score += 0.0
        else:
            score += -0.3

    # Free cash flow (positive = quality)
    fcf = safe_get(info, "freeCashflow")
    mcap = safe_get(info, "marketCap")
    if fcf is not None and mcap and mcap > 0:
        fcf_yield = fcf / mcap
        score += np.clip(fcf_yield / 0.05, -0.3, 0.3)

    return float(np.clip((score + 1) / 2, 0, 1))


# ── PROFITABILITY SCORER ──────────────────────────────────────────────────────

def profitability_score(info: dict) -> float:
    """
    Margin quality + return on capital.
    """
    score = 0.0

    # Gross margin
    gm = safe_get(info, "grossMargins")
    if gm is not None:
        score += np.clip(gm / 0.40, -0.5, 0.5) * 0.30

    # Operating margin
    om = safe_get(info, "operatingMargins")
    if om is not None:
        score += np.clip(om / 0.15, -0.5, 0.5) * 0.30

    # Return on equity
    roe = safe_get(info, "returnOnEquity")
    if roe is not None:
        score += np.clip(roe / 0.20, -0.5, 0.5) * 0.20

    # Return on assets
    roa = safe_get(info, "returnOnAssets")
    if roa is not None:
        score += np.clip(roa / 0.10, -0.5, 0.5) * 0.20

    return float(np.clip((score + 1) / 2, 0, 1))


# ── VALUATION SCORER ──────────────────────────────────────────────────────────

def valuation_score(info: dict) -> float:
    """
    NOT value investing — we're NOT hunting cheap stocks.
    We penalize extreme overvaluation (parabolic multiples)
    and reward reasonable growth-adjusted valuation.
    PEG ratio is the primary signal.
    """
    score = 0.5  # neutral default

    # PEG ratio (price/earnings ÷ growth rate)
    # PEG < 1 = undervalued growth, PEG > 3 = stretched
    peg = safe_get(info, "pegRatio")
    if peg is not None and peg > 0:
        if peg < 0.5:
            score = 0.9    # exceptional value for growth
        elif peg < 1.0:
            score = 0.75
        elif peg < 1.5:
            score = 0.65
        elif peg < 2.0:
            score = 0.55
        elif peg < 3.0:
            score = 0.45
        elif peg < 5.0:
            score = 0.30
        else:
            score = 0.15   # extremely stretched

    # P/S ratio override for unprofitable growth companies
    ps = safe_get(info, "priceToSalesTrailing12Months")
    if ps is not None and peg is None:
        if ps < 2:
            score = 0.80
        elif ps < 5:
            score = 0.65
        elif ps < 10:
            score = 0.50
        elif ps < 20:
            score = 0.35
        else:
            score = 0.20

    return float(np.clip(score, 0, 1))


# ── MAIN SCORER ───────────────────────────────────────────────────────────────

def score(universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score all tickers for fundamental quality.
    """
    symbols = universe_df["symbol"].tolist()
    rows    = []
    n       = len(symbols)

    log.info(f"  Fundamentals: scoring {n} symbols...")

    for i, sym in enumerate(symbols):
        if i % 100 == 0:
            log.info(f"    {i}/{n}")

        row = {"symbol": sym}

        try:
            t    = yf.Ticker(sym)
            info = t.info or {}

            # Quarterly financials for growth acceleration
            try:
                fin = t.quarterly_financials
            except Exception:
                fin = None

            grw  = growth_score(info, fin)
            qual = quality_score(info)
            prof = profitability_score(info)
            val  = valuation_score(info)

            # Composite: growth + quality are primary; value is a modifier
            composite = (
                grw  * 0.40 +
                qual * 0.25 +
                prof * 0.25 +
                val  * 0.10
            )

            row.update({
                "sig_fund_growth":         round(grw,       4),
                "sig_fund_quality":        round(qual,      4),
                "sig_fund_profitability":  round(prof,      4),
                "sig_fund_value":          round(val,       4),
                "sig_fundamentals":        round(composite, 4),
            })

        except Exception as e:
            log.debug(f"    {sym} fundamentals error: {e}")
            row.update({
                "sig_fund_growth":        np.nan,
                "sig_fund_quality":       np.nan,
                "sig_fund_profitability": np.nan,
                "sig_fund_value":         np.nan,
                "sig_fundamentals":       np.nan,
            })

        rows.append(row)

    scores_df = pd.DataFrame(rows)
    return universe_df.merge(scores_df, on="symbol", how="left")
