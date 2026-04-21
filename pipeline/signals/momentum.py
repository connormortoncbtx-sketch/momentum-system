"""
signals/momentum.py
====================
Momentum + technicals scorer.
Scores every ticker on pure price/volume behavior.

Output columns (all 0.0 – 1.0 normalized):
    sig_momentum_rs         Relative strength rank vs universe
    sig_momentum_trend      Trend quality (SMA stack + slope)
    sig_momentum_vol_surge  Volume confirmation
    sig_momentum_breakout   Near 52-week high breakout proximity
    sig_momentum            Composite momentum score
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


LOOKBACK = 252   # TARGET trading days of history -- must cover 12-1 month calc
# Calendar window needed: 252 trading days ≈ 252 × 7/5 = 353 calendar days.
# Plus a 30-day weekend/holiday buffer = 383. Previously we did `days + 30`
# literally (282 calendar days ≈ 201 trading days) and the 12-1 month return
# branch was UNREACHABLE because it requires len(close) >= 252. Result: r_12_1
# was always 0.0 for every ticker, weighted 40% in rs_return. That's the single
# strongest momentum horizon in the academic literature silently disabled.
_CAL_PER_TRADING_DAY = 7 / 5
_CAL_LOOKBACK_DAYS   = int(LOOKBACK * _CAL_PER_TRADING_DAY) + 30


import logging
import time

log = logging.getLogger(__name__)


# ── THROTTLING + RETRY CONFIG ─────────────────────────────────────────────────
# yfinance is a free-tier wrapper over Yahoo's public endpoints. Rate limiting
# is aggressive; a hammered endpoint returns empty data silently (no exception,
# no rate-limit error -- just empty frames). The 2026-04 pipeline runs showed
# 40%+ coverage gaps on market_cap and ~20% gaps on price history.
#
# Design choice: the Friday pipeline has no real-time deadline -- it just needs
# to finish before Monday morning. We trade runtime for coverage. Baseline
# measured at ~1 hour; throttled version should be ~1.5 hours but with far
# better data completeness.

FETCH_BATCH_SIZE       = 100   # down from 200 (smaller = less instantaneous load)
FETCH_BATCH_SLEEP      = 5.0   # inter-batch pause (was 0 in momentum/EV code)
FETCH_RETRY_WAIT       = 30.0  # pause before retrying a failed batch
FETCH_RETRY_WAIT_2     = 60.0  # pause before second retry
FETCH_MAX_RETRIES      = 2     # total of 3 attempts per batch (initial + 2 retries)


def _fetch_batch_with_retry(batch, start_str, end_str, label="batch"):
    """Download a batch from yfinance with retry-on-empty-or-exception.

    Returns the raw DataFrame (possibly empty) on success, None if all
    retries exhausted. An empty return from yfinance is treated as a soft
    failure and retried -- empty is the typical signature of rate-limiting
    on free-tier endpoints.
    """
    wait_schedule = [0, FETCH_RETRY_WAIT, FETCH_RETRY_WAIT_2]
    for attempt, wait_before in enumerate(wait_schedule):
        if wait_before > 0:
            log.info(f"    Retry attempt {attempt} after {wait_before:.0f}s wait...")
            time.sleep(wait_before)
        try:
            raw = yf.download(
                batch,
                start=start_str,
                end=end_str,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw is None or raw.empty:
                continue  # empty response -> treat as rate-limit, retry
            # Quick sanity: at least one non-null close price somewhere
            if isinstance(raw.columns, pd.MultiIndex) and "Close" in raw.columns.get_level_values(0):
                if not raw["Close"].notna().any().any():
                    continue
            elif "Close" in raw.columns and not raw["Close"].notna().any():
                continue
            return raw
        except Exception as e:
            log.warning(f"    {label} attempt {attempt} error: {e}")
            continue
    log.warning(f"    {label}: all {len(wait_schedule)} attempts failed")
    return None


def fetch_history(symbols: list[str], days: int = LOOKBACK) -> dict[str, pd.DataFrame]:
    """
    Download full OHLCV history for all symbols in batches.
    Returns {symbol: DataFrame(open,high,low,close,volume)}.

    `days` param retained as trading-day target for back-compat; internally
    converts to a sufficient calendar window.

    Throttled + retried: batches use a 5s inter-batch pause and retry up to
    2 times on empty/exception response (treating empty as rate-limiting).
    """
    end   = datetime.today()
    # Convert requested trading days -> calendar window. If caller passed the
    # default (LOOKBACK), use the pre-computed _CAL_LOOKBACK_DAYS constant;
    # otherwise apply the same conversion.
    cal_days = _CAL_LOOKBACK_DAYS if days == LOOKBACK else int(days * _CAL_PER_TRADING_DAY) + 30
    start = end - timedelta(days=cal_days)
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")
    result = {}

    n_batches = (len(symbols) + FETCH_BATCH_SIZE - 1) // FETCH_BATCH_SIZE
    log.info(f"  Momentum fetch: {len(symbols):,} symbols in {n_batches} batches "
             f"(batch={FETCH_BATCH_SIZE}, sleep={FETCH_BATCH_SLEEP}s)")

    for batch_idx, i in enumerate(range(0, len(symbols), FETCH_BATCH_SIZE), start=1):
        batch = symbols[i:i+FETCH_BATCH_SIZE]

        if batch_idx % 10 == 0:
            log.info(f"    Batch {batch_idx}/{n_batches} ({len(result):,} tickers captured so far)")

        raw = _fetch_batch_with_retry(batch, start_str, end_str, label=f"batch-{batch_idx}")

        if raw is not None:
            if len(batch) == 1:
                sym = batch[0]
                df  = raw.rename(columns=str.lower)
                if not df.empty:
                    result[sym] = df
            else:
                for sym in batch:
                    try:
                        df = pd.DataFrame({
                            "open":   raw["Open"][sym],
                            "high":   raw["High"][sym],
                            "low":    raw["Low"][sym],
                            "close":  raw["Close"][sym],
                            "volume": raw["Volume"][sym],
                        }).dropna()
                        if len(df) >= 60:
                            result[sym] = df
                    except Exception:
                        continue
        # else: batch failed all retries; all symbols in batch get no data

        # Inter-batch pause (not after last batch)
        if batch_idx < n_batches:
            time.sleep(FETCH_BATCH_SLEEP)

    log.info(f"  Momentum fetch complete: {len(result):,}/{len(symbols):,} "
             f"tickers have usable history ({100*len(result)/max(1,len(symbols)):.1f}%)")
    return result


# ── INDIVIDUAL SIGNAL CALCS ───────────────────────────────────────────────────

def rs_return(close: pd.Series) -> float:
    """
    Weighted momentum return: 12-1 month return with recent tilt.
    Excludes last month (avoids short-term reversal contamination).
    """
    if len(close) < 200:
        return 0.0
    r_12_1 = (close.iloc[-21] / close.iloc[-252]) - 1 if len(close) >= 252 else 0.0
    r_6_1  = (close.iloc[-21] / close.iloc[-126]) - 1 if len(close) >= 126 else 0.0
    r_3_1  = (close.iloc[-21] / close.iloc[-63])  - 1 if len(close) >= 63  else 0.0
    # Weight: 40% 12-1m, 35% 6-1m, 25% 3-1m
    return 0.40 * r_12_1 + 0.35 * r_6_1 + 0.25 * r_3_1


def trend_score(close: pd.Series) -> float:
    """
    SMA stack quality + slope.
    Perfect uptrend: close > SMA20 > SMA50 > SMA200, all rising.
    """
    if len(close) < 200:
        return 0.0

    sma20  = close.rolling(20).mean().iloc[-1]
    sma50  = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    price  = close.iloc[-1]

    score = 0.0
    score += 0.25 if price  > sma20  else -0.25
    score += 0.25 if sma20  > sma50  else -0.25
    score += 0.25 if sma50  > sma200 else -0.25

    # Slope of SMA50 over last 20 days
    sma50_series = close.rolling(50).mean().dropna()
    if len(sma50_series) >= 20:
        y  = sma50_series.tail(20).values
        x  = np.arange(20)
        m  = np.polyfit(x, y, 1)[0]
        normalized_slope = m / sma200 if sma200 > 0 else 0
        score += np.clip(normalized_slope * 500, -0.25, 0.25)

    return float(np.clip(score, -1, 1))


def volume_surge(close: pd.Series, volume: pd.Series) -> float:
    """
    Recent volume vs 50-day avg, weighted by price direction.
    Strong up-volume = positive signal.
    """
    if len(volume) < 50:
        return 0.0

    vol_avg  = volume.rolling(50).mean().iloc[-1]
    vol_now  = volume.tail(5).mean()  # 5-day avg
    vol_ratio = (vol_now / vol_avg) - 1 if vol_avg > 0 else 0

    # Weight by price direction last 5 days
    price_dir = 1 if close.iloc[-1] > close.iloc[-5] else -1

    return float(np.clip(vol_ratio * price_dir, -1, 1))


def breakout_score(close: pd.Series, high: pd.Series) -> float:
    """
    Proximity to 52-week high. Stocks near highs have momentum.
    Also rewards clean breakouts above prior resistance.
    """
    if len(close) < 252:
        if len(close) < 60:
            return 0.0
        period = len(close)
    else:
        period = 252

    high_52w  = high.tail(period).max()
    price     = close.iloc[-1]
    pct_off   = (price / high_52w) - 1  # 0 = at high, -0.20 = 20% off high

    # Bonus for actual breakout (within 2% of 52w high)
    if pct_off >= -0.02:
        return 1.0
    elif pct_off >= -0.05:
        return 0.80
    elif pct_off >= -0.10:
        return 0.50
    elif pct_off >= -0.15:
        return 0.20
    elif pct_off >= -0.25:
        return -0.10
    else:
        return float(np.clip(pct_off * 3, -1, 0))


def atr_normalized_return(close: pd.Series, high: pd.Series, low: pd.Series,
                           n: int = 20) -> float:
    """Recent return normalized by ATR — rewards clean moves, penalizes choppy ones."""
    if len(close) < n + 1:
        return 0.0
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    ret = (close.iloc[-1] / close.iloc[-n]) - 1
    return float(np.clip((ret / atr) * close.iloc[-1] / n, -1, 1)) if atr > 0 else 0.0


# ── MAIN SCORER ───────────────────────────────────────────────────────────────

def score(universe_df: pd.DataFrame, history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Score all tickers in universe_df.
    Returns DataFrame with momentum signal columns added.
    """
    symbols = universe_df["symbol"].tolist()
    rows    = []

    # First pass: compute raw RS return for ranking
    rs_raw = {}
    for sym in symbols:
        if sym not in history:
            rs_raw[sym] = np.nan
            continue
        c = history[sym]["close"]
        rs_raw[sym] = rs_return(c)

    rs_series = pd.Series(rs_raw).dropna()
    # Convert to percentile rank 0-1
    rs_ranks  = rs_series.rank(pct=True)

    for sym in symbols:
        h = history.get(sym)
        row = {"symbol": sym}

        if h is None or len(h) < 60:
            row.update({
                "sig_momentum_rs":        np.nan,
                "sig_momentum_trend":     np.nan,
                "sig_momentum_vol_surge": np.nan,
                "sig_momentum_breakout":  np.nan,
                "sig_momentum":           np.nan,
            })
            rows.append(row)
            continue

        c = h["close"]
        v = h["volume"]
        hi = h["high"]
        lo = h["low"]

        rs   = float(rs_ranks.get(sym, 0.5))    # 0-1 percentile
        trd  = (trend_score(c) + 1) / 2          # rescale -1/1 → 0/1
        vsrg = (volume_surge(c, v) + 1) / 2
        brk  = (breakout_score(c, hi) + 1) / 2
        atr  = (atr_normalized_return(c, hi, lo) + 1) / 2

        # Composite: RS is the anchor, rest are modifiers
        composite = (
            rs   * 0.35 +
            trd  * 0.25 +
            brk  * 0.20 +
            vsrg * 0.12 +
            atr  * 0.08
        )

        row.update({
            "sig_momentum_rs":        round(rs,        4),
            "sig_momentum_trend":     round(trd,       4),
            "sig_momentum_vol_surge": round(vsrg,      4),
            "sig_momentum_breakout":  round(brk,       4),
            "sig_momentum":           round(composite, 4),
        })
        rows.append(row)

    scores_df = pd.DataFrame(rows)
    return universe_df.merge(scores_df, on="symbol", how="left")
