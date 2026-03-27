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


LOOKBACK = 252   # trading days of history to fetch


def fetch_history(symbols: list[str], days: int = LOOKBACK) -> dict[str, pd.DataFrame]:
    """
    Download full OHLCV history for all symbols in batches.
    Returns {symbol: DataFrame(open,high,low,close,volume)}.
    """
    end   = datetime.today()
    start = end - timedelta(days=days + 30)
    batch_size = 200
    result = {}

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            raw = yf.download(
                batch,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True,
            )
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
        except Exception:
            continue

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
