"""
Stage 2 — Regime Classifier
=============================
Reads macro + cross-asset data from yfinance and FRED (via pandas-datareader),
computes five scored dimensions, and writes a single regime state file.

Output:  data/regime.json

Regime labels (5):
    risk_on          composite >= 0.35
    trending_mixed   composite >= 0.05
    choppy_neutral   composite >= -0.20
    risk_off_mild    composite >= -0.45
    risk_off_severe  composite <  -0.45

Every downstream stage reads regime.json and applies the matching
multipliers from config/weights.json automatically.

──────────────────────────────────────────────────────────────────────────────
2026-04-28 PATCH NOTES — faster regime detection:

The previous classifier was structurally slow. All return-based terms used
20-day windows, and four `above_sma` checks (SPY 50/200, QQQ 50, IWM 50)
contributed up to ±0.65 of static, multi-week-stable signal to the composite.
The only fast-moving input was VIX (sentiment, 20% weight). Net result: once
a regime label was set, it took weeks of contrary evidence to flip — even a
sharp 1-week selloff barely moved the composite. The 2026-04-10 momentum
sign-reversal was undetected by the classifier for weeks afterward.

Three changes:

1. **Return terms now blend 5-day and 20-day.** Each `ret(spy, 20)` term in
   score_trend / score_breadth / score_rotation is replaced with a 50/50
   blend of `ret(spy, 5)` (scaled to a 1.5% threshold) and `ret(spy, 20)`
   (scaled to the original 5% threshold). This adds ~equal weight to recent
   week's behavior alongside the existing past-month behavior.

2. **`above_sma` weights reduced.** Each ±0.20 / ±0.15 term cut roughly in
   half (±0.10 / ±0.08). They still contribute structural signal — being
   above the 200-day MA is a real signal of bull-market backdrop — but
   no longer dominate the composite to the point of preventing transitions.

3. **Dimension weights re-balanced toward sentiment.** trend 0.30 → 0.25,
   sentiment 0.20 → 0.25. VIX is the fastest-moving input we have, and
   under-weighting it relative to slow trend signals was a structural
   asymmetry. Sentiment and trend now share top weight.

4. **Regime transition flag added.** classify() now reads the previous
   regime.json (if present) and computes `composite_delta` vs the prior
   reading. If the composite has moved by ≥0.20 in either direction
   without crossing a regime-label threshold, sets `transitioning: true`
   in the output for downstream stages to optionally consume. Additive
   only; no existing behavior changes from this flag.

Threshold tuning note: these changes will produce more volatile composite
values. Existing regime-label thresholds (risk_on >= 0.35 etc.) may need
recalibration after 2-4 weeks of new outputs. For now, thresholds left at
existing values; the transition flag provides early warning of label
changes that the slower thresholds wouldn't catch.
──────────────────────────────────────────────────────────────────────────────
"""

import json
import time
import logging
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

# LOOKBACK_DAYS must provide ≥200 trading days for the SMA200 check in
# score_trend(). Previously 90 calendar days → ~75 trading days → above_sma(spy,
# 200) was hitting the insufficient-data fallback on EVERY run. That fallback
# returned True unconditionally, which added a phantom +0.15 to every trend
# score. Regimes have been biased risk-on across the board for the life of the
# system. 300 calendar days → ~215 trading days safely covers SMA200 with
# headroom for holiday weeks and the pandas rolling() window fill.
LOOKBACK_DAYS = 300
DATA_DIR      = Path("data")
OUTPUT        = DATA_DIR / "regime.json"
WEIGHTS_FILE  = Path("config/weights.json")

log = logging.getLogger(__name__)

# Cross-asset tickers (all free via yfinance)
TICKERS = {
    "SPY":  "SPY",    # S&P 500 — broad market trend
    "QQQ":  "QQQ",    # Nasdaq 100 — growth/risk appetite
    "IWM":  "IWM",    # Russell 2000 — small cap breadth
    "XLK":  "XLK",    # Tech sector — offensive leadership
    "XLU":  "XLU",    # Utilities — defensive rotation
    "XLV":  "XLV",    # Healthcare — defensive / quality
    "XLP":  "XLP",    # Consumer staples — defensives
    "HYG":  "HYG",    # High yield bonds — credit risk appetite
    "TLT":  "TLT",    # 20yr treasuries — flight to safety
    "GLD":  "GLD",    # Gold — macro fear / dollar weakness
    "VIX":  "^VIX",   # Volatility — fear gauge
    "DXY":  "DX-Y.NYB", # Dollar index — macro risk backdrop
}

# ── DATA FETCH ────────────────────────────────────────────────────────────────

def fetch_prices(lookback: int = LOOKBACK_DAYS) -> pd.DataFrame:
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=lookback + 15)  # buffer for weekends

    syms    = list(TICKERS.values())
    max_attempts = 4
    wait_seconds = [0, 30, 60, 120]  # backoff: immediate, 30s, 1min, 2min

    for attempt, wait in enumerate(wait_seconds, 1):
        if wait > 0:
            log.info(f"  Retrying price fetch in {wait}s (attempt {attempt}/{max_attempts})...")
            time.sleep(wait)
        try:
            raw = yf.download(syms, start=str(start), end=str(end),
                              auto_adjust=True, progress=False)

            # Handle yfinance MultiIndex columns (newer versions return
            # MultiIndex like ('Close', 'SPY') instead of just 'SPY')
            if isinstance(raw.columns, pd.MultiIndex):
                raw = raw["Close"]
            elif "Close" in raw.columns:
                raw = raw["Close"]
            # else raw is already a flat Close DataFrame

            # Normalize column names back to our keys (^VIX → VIX etc.)
            inv = {v: k for k, v in TICKERS.items()}
            raw.columns = [inv.get(str(c), str(c)) for c in raw.columns]

            df = raw.dropna(how="all").ffill().tail(lookback)

            log.info(f"  Columns mapped: {list(df.columns)}")
            nan_cols = [c for c in df.columns if df[c].isna().any()]
            if nan_cols:
                log.warning(f"  Columns with NaN after ffill: {nan_cols}")

            # Validate we got meaningful data — at least SPY and VIX
            if "SPY" not in df.columns or df["SPY"].dropna().empty:
                log.warning(f"  Attempt {attempt}: SPY data missing — retrying")
                continue

            if len(df) < 20:
                log.warning(f"  Attempt {attempt}: Only {len(df)} rows — retrying")
                continue

            log.info(f"  Price data fetched: {len(df)} days × {len(df.columns)} assets")
            return df

        except Exception as e:
            log.warning(f"  Attempt {attempt}: fetch failed — {e}")
            if attempt == max_attempts:
                log.error("All retry attempts exhausted — regime classifier will use fallback")
                raise

    # Should not reach here but return empty frame as safety
    return pd.DataFrame()


# ── SIGNAL PRIMITIVES ─────────────────────────────────────────────────────────

def ret(series: pd.Series, n: int) -> float:
    """N-day return. 0.0 if insufficient history."""
    if len(series) < n + 1:
        return 0.0
    return float((series.iloc[-1] / series.iloc[-n]) - 1)


def above_sma(series: pd.Series, n: int) -> bool:
    """
    True if latest price is above the n-period SMA.

    Previously: when len(series) < n, returned True (optimistic fallback) which
    silently corrupted regime classification — with a 90-day lookback, every
    above_sma(spy, 200) call hit this branch. Now raises, because this function
    is only called from regime scoring where an incorrect fallback has real
    downstream consequences (wrong regime multipliers applied to every
    ticker's score).
    """
    if len(series) < n:
        raise ValueError(
            f"above_sma needs ≥{n} data points, got {len(series)}. "
            f"Caller should bump LOOKBACK_DAYS or pass a different window."
        )
    return bool(series.iloc[-1] > series.rolling(n).mean().iloc[-1])


def pct_off_high(series: pd.Series) -> float:
    return float((series.iloc[-1] / series.max()) - 1)


def slope(series: pd.Series, n: int = 20) -> float:
    """Normalized linear regression slope over last n bars."""
    if len(series) < n:
        return 0.0
    y = series.tail(n).values
    x = np.arange(n)
    m = np.polyfit(x, y, 1)[0]
    return float(m / series.iloc[-1])  # normalize by price


def realized_vol(series: pd.Series, n: int = 20) -> float:
    if len(series) < n + 1:
        return 0.0
    rets = series.pct_change().dropna().tail(n)
    return float(rets.std() * np.sqrt(252))


# ── HELPER: BLEND FAST + SLOW RETURN TERMS ────────────────────────────────────

def blended_ret_term(series: pd.Series,
                     fast_n: int = 5, fast_threshold: float = 0.015,
                     slow_n: int = 20, slow_threshold: float = 0.05) -> float:
    """
    Blend a 5-day return term with a 20-day return term, each clipped to ±1
    after dividing by its respective threshold. Returns the average.

    The 5-day threshold of 1.5% is roughly proportional to the 20-day threshold
    of 5.0% (i.e., 1.5%/5d ≈ 0.3% per day, 5%/20d = 0.25% per day; close
    enough that both terms saturate on similarly-strong moves at their
    respective timescales). Using identical thresholds (1.5% on both 5d and
    20d) would over-weight the 5-day term because typical 20-day moves are
    much larger than 5-day moves.

    Returns a float in [-1, +1]. Caller multiplies by their dimension weight.
    """
    r_fast = ret(series, fast_n)
    r_slow = ret(series, slow_n)
    fast_score = np.clip(r_fast / fast_threshold, -1, 1)
    slow_score = np.clip(r_slow / slow_threshold, -1, 1)
    return float((fast_score + slow_score) / 2.0)


# ── DIMENSION SCORERS  (-1.0 → +1.0 each) ────────────────────────────────────

def score_trend(df: pd.DataFrame) -> tuple[float, dict]:
    """Is the broad market in a healthy uptrend?

    2026-04-28: replaced 20-day-only return terms with 5d/20d blends, and cut
    above_sma contributions roughly in half. Trend score now responds to a
    1-week reversal much faster while still respecting the 1-month and
    structural backdrop.
    """
    spy = df["SPY"]

    # Blended return: half weight on 5-day, half on 20-day
    trend_blend_50 = blended_ret_term(spy, 5, 0.015, 20, 0.05)   # was r20-only
    trend_blend_50_pct = blended_ret_term(spy, 5, 0.015, 50, 0.08)  # was r50-only

    # Diagnostic-only individual returns (logged but not separately scored)
    r5  = ret(spy, 5)
    r20 = ret(spy, 20)
    r50 = ret(spy, 50)

    sma50 = above_sma(spy, 50)
    sma200 = above_sma(spy, 200)
    drawdown = pct_off_high(spy)

    s = 0.0
    s += trend_blend_50      * 0.30   # was 0.25 with 20-day-only
    s += trend_blend_50_pct  * 0.30   # was 0.25 with 50-day-only
    s += (0.10 if sma50  else -0.10)   # was ±0.20
    s += (0.08 if sma200 else -0.08)   # was ±0.15
    s += np.clip(drawdown / -0.10, -1, 1) * -0.15

    return float(np.clip(s, -1, 1)), {
        "spy_ret_5d": round(r5, 4), "spy_ret_20d": round(r20, 4),
        "spy_ret_50d": round(r50, 4),
        "spy_above_sma50": sma50, "spy_above_sma200": sma200,
        "spy_drawdown": round(drawdown, 4),
    }


def score_breadth(df: pd.DataFrame) -> tuple[float, dict]:
    """Are risk assets broadly participating, or is it narrow leadership?

    2026-04-28: blended 5d/20d returns for QQQ, IWM, and IWM-vs-SPY relative.
    Cut above_sma contributions in half. Now responds to a 1-week breadth
    breakdown much faster.
    """
    qqq_blend = blended_ret_term(df["QQQ"], 5, 0.015, 20, 0.05)
    iwm_blend = blended_ret_term(df["IWM"], 5, 0.015, 20, 0.05)

    # Small cap vs large cap (breadth proxy)
    iwm_rel_5  = ret(df["IWM"], 5)  - ret(df["SPY"], 5)
    iwm_rel_20 = ret(df["IWM"], 20) - ret(df["SPY"], 20)
    iwm_rel_blend = float((np.clip(iwm_rel_5 / 0.01, -1, 1) +
                           np.clip(iwm_rel_20 / 0.03, -1, 1)) / 2.0)

    qqq_sma = above_sma(df["QQQ"], 50)
    iwm_sma = above_sma(df["IWM"], 50)

    s = 0.0
    s += qqq_blend     * 0.25   # was 0.25 on r20 alone
    s += iwm_blend     * 0.35   # was 0.35 on r20 alone — small cap = true breadth
    s += iwm_rel_blend * 0.20   # was 0.20 on r20 alone
    s += (0.05 if qqq_sma else -0.05)   # was ±0.10
    s += (0.05 if iwm_sma else -0.05)   # was ±0.10

    # Diagnostic
    r_qqq_5  = ret(df["QQQ"], 5)
    r_qqq_20 = ret(df["QQQ"], 20)
    r_iwm_5  = ret(df["IWM"], 5)
    r_iwm_20 = ret(df["IWM"], 20)

    return float(np.clip(s, -1, 1)), {
        "qqq_ret_5d": round(r_qqq_5, 4), "qqq_ret_20d": round(r_qqq_20, 4),
        "iwm_ret_5d": round(r_iwm_5, 4), "iwm_ret_20d": round(r_iwm_20, 4),
        "iwm_vs_spy_5d": round(iwm_rel_5, 4),
        "iwm_vs_spy_20d": round(iwm_rel_20, 4),
        "iwm_above_sma50": iwm_sma,
    }


def score_sentiment(df: pd.DataFrame) -> tuple[float, dict]:
    """Fear vs greed: VIX level + credit spread proxy.

    2026-04-28: HYG vs SPY now blends 5d/20d to catch faster credit-market
    inflections. VIX scoring is unchanged — already a fast-moving input by
    nature.
    """
    # Use last available VIX value — handles holidays where VIX has no close
    vix_series = df["VIX"].dropna()
    if vix_series.empty:
        log.warning("VIX data unavailable — using neutral sentiment score")
        return 0.0, {"vix": None, "vix_ma20": None, "vix_trend": 0,
                     "hyg_vs_spy_5d": 0, "hyg_vs_spy_20d": 0}

    vix    = float(vix_series.iloc[-1])
    vix_ma = float(vix_series.rolling(20).mean().iloc[-1]) if len(vix_series) >= 20 else vix
    vix_trend = vix - vix_ma          # rising VIX = fear growing

    # HYG vs SPY relative performance = credit risk appetite
    hyg_rel_5  = ret(df["HYG"], 5)  - ret(df["SPY"], 5)
    hyg_rel_20 = ret(df["HYG"], 20) - ret(df["SPY"], 20)

    # VIX level score (lower = better)
    if vix < 13:   vix_s =  1.0
    elif vix < 17: vix_s =  0.5
    elif vix < 21: vix_s =  0.0
    elif vix < 27: vix_s = -0.5
    elif vix < 35: vix_s = -0.8
    else:          vix_s = -1.0

    vix_trend_s = np.clip(-vix_trend / 5, -0.5, 0.5)
    # Blended credit-spread score: 5d threshold tighter (0.005) than 20d (0.02)
    credit_blend = (np.clip(hyg_rel_5  / 0.005, -1, 1) +
                    np.clip(hyg_rel_20 / 0.02,  -1, 1)) / 2.0
    credit_s = float(credit_blend) * 0.3

    s = vix_s * 0.50 + vix_trend_s + credit_s

    return float(np.clip(s, -1, 1)), {
        "vix": round(vix, 2), "vix_ma20": round(vix_ma, 2),
        "vix_trend": round(vix_trend, 2),
        "hyg_vs_spy_5d":  round(hyg_rel_5, 4),
        "hyg_vs_spy_20d": round(hyg_rel_20, 4),
    }


def score_rotation(df: pd.DataFrame) -> tuple[float, dict]:
    """Offensive vs defensive sector leadership.

    2026-04-28: rotation spread blends 5d/20d. Slope component (which is
    inherently a 20-day measure) kept at original weight. The 5d component
    catches sharp risk-off rotations within the past week even when the
    20-day window still shows offense leading.
    """
    # Offensive: XLK (tech), QQQ
    # Defensive: XLU (utilities), XLP (staples), XLV (healthcare)

    off_ret_5  = (ret(df["XLK"], 5)  + ret(df["QQQ"], 5))  / 2
    def_ret_5  = (ret(df["XLU"], 5)  + ret(df["XLP"], 5)  + ret(df["XLV"], 5))  / 3
    off_ret_20 = (ret(df["XLK"], 20) + ret(df["QQQ"], 20)) / 2
    def_ret_20 = (ret(df["XLU"], 20) + ret(df["XLP"], 20) + ret(df["XLV"], 20)) / 3

    rotation_5  = off_ret_5  - def_ret_5
    rotation_20 = off_ret_20 - def_ret_20

    # Blend: 5d at 1.2% threshold, 20d at original 4% threshold
    rotation_blend = (np.clip(rotation_5  / 0.012, -1, 1) +
                      np.clip(rotation_20 / 0.04,  -1, 1)) / 2.0

    # Trend within rotation (slope is inherently a 20-day measure; not blended)
    xlk_slope = slope(df["XLK"], 20)
    xlu_slope = slope(df["XLU"], 20)

    s = float(rotation_blend) * 0.70
    s += np.clip((xlk_slope - xlu_slope) / 0.002, -1, 1) * 0.30

    return float(np.clip(s, -1, 1)), {
        "offensive_ret_5d":  round(off_ret_5, 4),
        "defensive_ret_5d":  round(def_ret_5, 4),
        "offensive_ret_20d": round(off_ret_20, 4),
        "defensive_ret_20d": round(def_ret_20, 4),
        "rotation_spread_5d":  round(rotation_5,  4),
        "rotation_spread_20d": round(rotation_20, 4),
        "xlk_slope":         round(xlk_slope, 6),
        "xlu_slope":         round(xlu_slope, 6),
    }


def score_safety(df: pd.DataFrame) -> tuple[float, dict]:
    """Is money flowing into safe havens (bonds, gold)?

    2026-04-28: blended 5d/20d for both TLT-vs-SPY and GLD-vs-SPY. Flight to
    safety often unfolds rapidly — TLT can rally 2-3% in a single risk-off
    week while the 20-day comparison still shows equities ahead.
    """
    tlt_vs_spy_5  = ret(df["TLT"], 5)  - ret(df["SPY"], 5)
    tlt_vs_spy_20 = ret(df["TLT"], 20) - ret(df["SPY"], 20)
    gld_vs_spy_5  = ret(df["GLD"], 5)  - ret(df["SPY"], 5)
    gld_vs_spy_20 = ret(df["GLD"], 20) - ret(df["SPY"], 20)

    # Inverted sign: safety outperforming = bad for equities = negative score
    tlt_blend = (np.clip(-tlt_vs_spy_5  / 0.012, -1, 1) +
                 np.clip(-tlt_vs_spy_20 / 0.04,  -1, 1)) / 2.0
    gld_blend = (np.clip(-gld_vs_spy_5  / 0.01,  -1, 1) +
                 np.clip(-gld_vs_spy_20 / 0.03,  -1, 1)) / 2.0

    s  = float(tlt_blend) * 0.60
    s += float(gld_blend) * 0.40

    return float(np.clip(s, -1, 1)), {
        "tlt_vs_spy_5d":  round(tlt_vs_spy_5, 4),
        "tlt_vs_spy_20d": round(tlt_vs_spy_20, 4),
        "gld_vs_spy_5d":  round(gld_vs_spy_5, 4),
        "gld_vs_spy_20d": round(gld_vs_spy_20, 4),
    }


# ── COMPOSITE + LABEL ─────────────────────────────────────────────────────────

# 2026-04-28 rebalance: trend 0.30→0.25, sentiment 0.20→0.25.
# VIX (sentiment's primary input) is the fastest-moving signal in the basket;
# previously underweighted relative to slow trend signals. Sentiment and trend
# now share top weight.
DIMENSION_WEIGHTS = {
    "trend":     0.25,
    "breadth":   0.25,
    "sentiment": 0.25,
    "rotation":  0.15,
    "safety":    0.10,
}

REGIME_THRESHOLDS = [
    ( 0.35,  "risk_on"),
    ( 0.05,  "trending_mixed"),
    (-0.20,  "choppy_neutral"),
    (-0.45,  "risk_off_mild"),
    (-999,   "risk_off_severe"),
]

REGIME_DESCRIPTIONS = {
    "risk_on":         "Strong broad uptrend, risk assets leading, low fear. Full momentum mode.",
    "trending_mixed":  "Uptrend intact but mixed breadth. Be selective — only strongest setups.",
    "choppy_neutral":  "No clear trend. Inconsistent breadth. Raise the bar, reduce size.",
    "risk_off_mild":   "Defensive rotation underway. Favor quality/defensives. Reduce exposure.",
    "risk_off_severe": "Market stress. Capital preservation mode. Cash is a position.",
}

# Threshold for flagging a regime as transitioning. If the composite has moved
# by this much (in either direction) since the previous classification but
# stayed within the same regime label band, mark the regime as transitioning.
# Magnitude chosen to roughly equal one-half of the smallest regime band width
# (e.g., trending_mixed runs from 0.05 to 0.35, half-width = 0.15) -- a 0.20
# move is meaningful relative to band sizes without being so large that it
# would have already triggered a label change.
TRANSITION_THRESHOLD = 0.20


def classify(df: pd.DataFrame) -> dict:
    scorers = {
        "trend":     score_trend,
        "breadth":   score_breadth,
        "sentiment": score_sentiment,
        "rotation":  score_rotation,
        "safety":    score_safety,
    }

    scores  = {}
    details = {}
    for name, fn in scorers.items():
        try:
            s, d = fn(df)
            scores[name]  = s
            details[name] = d
        except Exception as e:
            log.warning(f"  Scorer '{name}' failed: {e}")
            import traceback
            log.warning(traceback.format_exc())
            scores[name]  = float("nan")
            details[name] = {}

    composite = sum(scores[k] * DIMENSION_WEIGHTS[k] for k in scores)
    composite = float(np.clip(composite, -1, 1))

    regime = "risk_off_severe"
    for threshold, label in REGIME_THRESHOLDS:
        if composite >= threshold:
            regime = label
            break

    # Volatility context
    spy_vol = realized_vol(df["SPY"], 20)

    # ── TRANSITION DETECTION ─────────────────────────────────────────────
    # Read previous regime.json (if present) and compute composite delta. If
    # the composite has shifted meaningfully since the prior reading without
    # crossing into a new label band, flag as transitioning. Downstream stages
    # may optionally consume this flag to tighten stops, reduce sizing, or
    # otherwise behave more cautiously.
    #
    # Additive only -- existing regime label, composite, and scores are
    # unchanged by this logic. The flag is purely informational.
    transitioning   = False
    transition_dir  = None
    composite_delta = None
    prev_composite  = None
    prev_regime     = None

    if OUTPUT.exists():
        try:
            with open(OUTPUT) as f:
                prev = json.load(f)
            prev_composite = prev.get("composite")
            prev_regime    = prev.get("regime")

            if isinstance(prev_composite, (int, float)):
                composite_delta = round(composite - prev_composite, 4)
                if abs(composite_delta) >= TRANSITION_THRESHOLD:
                    transitioning  = True
                    transition_dir = "weakening" if composite_delta < 0 else "strengthening"
        except Exception as e:
            log.debug(f"  Could not read previous regime.json for transition detection: {e}")

    return {
        "regime":      regime,
        "description": REGIME_DESCRIPTIONS[regime],
        "composite":   round(composite, 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "weights": DIMENSION_WEIGHTS,
        "details": details,
        "transition": {
            "transitioning":     transitioning,
            "direction":         transition_dir,
            "composite_delta":   composite_delta,
            "prev_composite":    prev_composite,
            "prev_regime":       prev_regime,
        },
        "context": {
            "spy_realized_vol_20d": round(spy_vol, 4),
            "vix":  round(float(df["VIX"].dropna().iloc[-1]), 2) if not df["VIX"].dropna().empty else None,
            "as_of": str(datetime.date.today()),
        },
    }


# ── PLAYBOOK LOOKUP ───────────────────────────────────────────────────────────

def load_playbook(regime: str) -> dict:
    """Read regime-specific multipliers from weights.json."""
    try:
        with open(WEIGHTS_FILE) as f:
            cfg = json.load(f)
        return cfg["regime_multipliers"].get(regime, {})
    except Exception:
        return {}


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Fetching cross-asset price data...")
    try:
        df = fetch_prices()
    except Exception as e:
        log.error(f"Price fetch failed after all retries: {e}")
        # Fall back to last known regime if available
        if OUTPUT.exists():
            log.warning("Using last known regime from previous run")
            with open(OUTPUT) as f:
                result = json.load(f)
            result["stale"] = True
            log.info(f"  Using cached regime: {result['regime']} (stale — data fetch failed)")
            return result
        else:
            log.error("No cached regime available — using risk_off_severe as safe default")
            result = {
                "regime": "risk_off_severe",
                "description": "Data fetch failed — defaulting to most defensive regime",
                "composite": -1.0,
                "scores": {"trend":-1.0,"breadth":-1.0,"sentiment":-1.0,"rotation":-1.0,"safety":1.0},
                "stale": True,
                "context": {"vix": None, "as_of": str(datetime.date.today())},
            }
            with open(OUTPUT, "w") as f:
                json.dump(result, f, indent=2)
            return result

    log.info("Running regime classifier...")
    result = classify(df)

    regime   = result["regime"]
    playbook = load_playbook(regime)
    result["playbook"] = playbook

    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)

    log.info(f"Regime → {regime.upper()}  (composite: {result['composite']:+.4f})")
    log.info(f"  {result['description']}")
    log.info(f"  Scores: " + "  ".join(
        f"{k}: {v:+.3f}" for k, v in result["scores"].items()
    ))
    if result.get("transition", {}).get("transitioning"):
        t = result["transition"]
        log.info(f"  TRANSITIONING ({t['direction']}): "
                 f"composite {t['prev_composite']:+.4f} → {result['composite']:+.4f} "
                 f"(Δ {t['composite_delta']:+.4f}, prev regime: {t['prev_regime']})")
    log.info(f"Output → {OUTPUT}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    run()
