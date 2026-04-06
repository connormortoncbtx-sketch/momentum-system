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

LOOKBACK_DAYS = 90
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
                              auto_adjust=True, progress=False)["Close"]

            # Normalize column names back to our keys
            inv = {v: k for k, v in TICKERS.items()}
            raw.columns = [inv.get(c, c) for c in raw.columns]

            df = raw.dropna(how="all").tail(lookback)

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
    if len(series) < n:
        return True
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


# ── DIMENSION SCORERS  (-1.0 → +1.0 each) ────────────────────────────────────

def score_trend(df: pd.DataFrame) -> tuple[float, dict]:
    """Is the broad market in a healthy uptrend?"""
    spy = df["SPY"]

    r20  = ret(spy, 20)
    r50  = ret(spy, 50)
    sma50 = above_sma(spy, 50)
    sma200 = above_sma(spy, 200)
    drawdown = pct_off_high(spy)
    slp  = slope(spy, 20)

    s = 0.0
    s += np.clip(r20  / 0.05, -1, 1) * 0.25
    s += np.clip(r50  / 0.08, -1, 1) * 0.25
    s += (0.20 if sma50  else -0.20)
    s += (0.15 if sma200 else -0.15)
    s += np.clip(drawdown / -0.10, -1, 1) * -0.15

    return float(np.clip(s, -1, 1)), {
        "spy_ret_20d": round(r20, 4), "spy_ret_50d": round(r50, 4),
        "spy_above_sma50": sma50, "spy_above_sma200": sma200,
        "spy_drawdown": round(drawdown, 4),
    }


def score_breadth(df: pd.DataFrame) -> tuple[float, dict]:
    """Are risk assets broadly participating, or is it narrow leadership?"""
    r_qqq = ret(df["QQQ"], 20)
    r_iwm = ret(df["IWM"], 20)
    r_spy = ret(df["SPY"], 20)

    # Small cap vs large cap (breadth proxy)
    iwm_rel = r_iwm - r_spy
    qqq_rel = r_qqq - r_spy

    # Are both above their 50-day SMAs?
    qqq_sma = above_sma(df["QQQ"], 50)
    iwm_sma = above_sma(df["IWM"], 50)

    s = 0.0
    s += np.clip(r_qqq / 0.05, -1, 1) * 0.25
    s += np.clip(r_iwm / 0.05, -1, 1) * 0.35   # small cap = true breadth
    s += np.clip(iwm_rel / 0.03, -1, 1) * 0.20
    s += (0.10 if qqq_sma else -0.10)
    s += (0.10 if iwm_sma else -0.10)

    return float(np.clip(s, -1, 1)), {
        "qqq_ret_20d": round(r_qqq, 4), "iwm_ret_20d": round(r_iwm, 4),
        "iwm_vs_spy": round(iwm_rel, 4), "iwm_above_sma50": iwm_sma,
    }


def score_sentiment(df: pd.DataFrame) -> tuple[float, dict]:
    """Fear vs greed: VIX level + credit spread proxy."""
    # Use last available VIX value — handles holidays where VIX has no close
    vix_series = df["VIX"].dropna()
    if vix_series.empty:
        log.warning("VIX data unavailable — using neutral sentiment score")
        return 0.0, {"vix": None, "vix_ma20": None, "vix_trend": 0, "hyg_vs_spy_20d": 0}

    vix    = float(vix_series.iloc[-1])
    vix_ma = float(vix_series.rolling(20).mean().iloc[-1]) if len(vix_series) >= 20 else vix
    vix_trend = vix - vix_ma          # rising VIX = fear growing

    # HYG vs SPY relative performance = credit risk appetite
    hyg_rel = ret(df["HYG"], 20) - ret(df["SPY"], 20)

    # VIX level score (lower = better)
    if vix < 13:   vix_s =  1.0
    elif vix < 17: vix_s =  0.5
    elif vix < 21: vix_s =  0.0
    elif vix < 27: vix_s = -0.5
    elif vix < 35: vix_s = -0.8
    else:          vix_s = -1.0

    vix_trend_s = np.clip(-vix_trend / 5, -0.5, 0.5)
    credit_s    = np.clip(hyg_rel / 0.02, -1, 1) * 0.3

    s = vix_s * 0.50 + vix_trend_s + credit_s

    return float(np.clip(s, -1, 1)), {
        "vix": round(vix, 2), "vix_ma20": round(vix_ma, 2),
        "vix_trend": round(vix_trend, 2), "hyg_vs_spy_20d": round(hyg_rel, 4),
    }


def score_rotation(df: pd.DataFrame) -> tuple[float, dict]:
    """Offensive vs defensive sector leadership."""
    # Offensive: XLK (tech), QQQ
    # Defensive: XLU (utilities), XLP (staples), XLV (healthcare)

    off_ret  = (ret(df["XLK"], 20) + ret(df["QQQ"], 20)) / 2
    def_ret  = (ret(df["XLU"], 20) + ret(df["XLP"], 20) + ret(df["XLV"], 20)) / 3
    rotation = off_ret - def_ret

    # Trend within rotation
    xlk_slope = slope(df["XLK"], 20)
    xlu_slope = slope(df["XLU"], 20)

    s = np.clip(rotation / 0.04, -1, 1) * 0.70
    s += np.clip((xlk_slope - xlu_slope) / 0.002, -1, 1) * 0.30

    return float(np.clip(s, -1, 1)), {
        "offensive_ret_20d": round(off_ret, 4),
        "defensive_ret_20d": round(def_ret, 4),
        "rotation_spread":   round(rotation, 4),
        "xlk_slope":         round(xlk_slope, 6),
        "xlu_slope":         round(xlu_slope, 6),
    }


def score_safety(df: pd.DataFrame) -> tuple[float, dict]:
    """Is money flowing into safe havens (bonds, gold)?"""
    tlt_ret = ret(df["TLT"], 20)
    gld_ret = ret(df["GLD"], 20)
    spy_ret = ret(df["SPY"], 20)

    # Bonds and gold rallying vs equities = flight to safety = bearish
    tlt_vs_spy = tlt_ret - spy_ret
    gld_vs_spy = gld_ret - spy_ret

    # Inverted: safety outperforming is bad for equities
    s  = np.clip(-tlt_vs_spy / 0.04, -1, 1) * 0.60
    s += np.clip(-gld_vs_spy / 0.03, -1, 1) * 0.40

    return float(np.clip(s, -1, 1)), {
        "tlt_ret_20d": round(tlt_ret, 4),
        "gld_ret_20d": round(gld_ret, 4),
        "tlt_vs_spy":  round(tlt_vs_spy, 4),
        "gld_vs_spy":  round(gld_vs_spy, 4),
    }


# ── COMPOSITE + LABEL ─────────────────────────────────────────────────────────

DIMENSION_WEIGHTS = {
    "trend":     0.30,
    "breadth":   0.25,
    "sentiment": 0.20,
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
        s, d = fn(df)
        scores[name]  = s
        details[name] = d

    composite = sum(scores[k] * DIMENSION_WEIGHTS[k] for k in scores)
    composite = float(np.clip(composite, -1, 1))

    regime = "risk_off_severe"
    for threshold, label in REGIME_THRESHOLDS:
        if composite >= threshold:
            regime = label
            break

    # Volatility context
    spy_vol = realized_vol(df["SPY"], 20)

    return {
        "regime":      regime,
        "description": REGIME_DESCRIPTIONS[regime],
        "composite":   round(composite, 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "weights": DIMENSION_WEIGHTS,
        "details": details,
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
    log.info(f"Output → {OUTPUT}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    run()
