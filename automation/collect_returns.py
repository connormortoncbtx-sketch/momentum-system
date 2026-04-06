"""
automation/collect_returns.py
==============================
Runs every Friday at 6pm CT (before the pipeline).

Collects the full weekly price curve for every scored ticker so the
learning loop can analyze not just WHAT to buy but WHEN to enter and exit.

For each ticker each week we capture:
    - Friday prior close (entry baseline)
    - Monday open, high, low, close
    - Tuesday open, high, low, close
    - Wednesday open, high, low, close
    - Thursday open, high, low, close
    - Friday open, high, low, close
    - Weekly high price + day it occurred
    - Weekly low price + day it occurred

Derived return columns:
    return_fri_fri      Friday close to Friday close (full theoretical week)
    return_mon_open     Monday open to Friday close
    return_tue_open     Tuesday open to Friday close  <- primary entry window
    return_wed_open     Wednesday open to Friday close
    return_fri_mon      Friday close to Monday close (Monday-only move)
    return_mon_tue      Monday close to Tuesday open (gap context)
    return_mon_peak     Monday open to weekly high (best case Monday entry)
    return_tue_peak     Tuesday open to weekly high (best case Tuesday entry)

Over time self_refine.py analyzes which entry day produces best avg return
by regime, signal strength, and pre-market conditions.
"""

import logging
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from automation.tz_utils import now_ct, assert_normal_week

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")

DATA_DIR  = Path("data")
PERF_LOG  = DATA_DIR / "performance_log.csv"
SCORES    = DATA_DIR / "scores_final.csv"
BATCH_SIZE = 150


def last_friday(from_date=None) -> datetime:
    d = from_date or datetime.today()
    days_back = (d.weekday() - 4) % 7
    return (d - timedelta(days=days_back)).replace(
        hour=0, minute=0, second=0, microsecond=0)


def trading_days_since(from_date: datetime) -> int:
    today = datetime.today()
    if from_date >= today:
        return 0
    days = 0
    current = from_date + timedelta(days=1)
    while current <= today:
        if current.weekday() < 5:
            days += 1
        current += timedelta(days=1)
    return days


def get_week_dates(week_start_friday: datetime) -> dict:
    monday = week_start_friday + timedelta(days=3)
    return {
        "mon": monday,
        "tue": monday + timedelta(days=1),
        "wed": monday + timedelta(days=2),
        "thu": monday + timedelta(days=3),
        "fri": monday + timedelta(days=4),
    }


def fetch_weekly_ohlcv(symbols, week_start_friday):
    dates    = get_week_dates(week_start_friday)
    dl_start = (week_start_friday - timedelta(days=1)).strftime("%Y-%m-%d")
    dl_end   = (dates["fri"] + timedelta(days=2)).strftime("%Y-%m-%d")

    log.info(f"Fetching OHLCV {dl_start} to {dl_end} for {len(symbols):,} symbols...")
    result = {}

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        if i % 500 == 0 and i > 0:
            log.info(f"  Progress: {i}/{len(symbols)}")
        try:
            raw = yf.download(batch, start=dl_start, end=dl_end,
                              auto_adjust=True, progress=False, threads=True)

            if isinstance(raw.columns, pd.MultiIndex):
                for sym in batch:
                    try:
                        df = pd.DataFrame({
                            "open":   raw["Open"][sym],
                            "high":   raw["High"][sym],
                            "low":    raw["Low"][sym],
                            "close":  raw["Close"][sym],
                            "volume": raw["Volume"][sym],
                        }).dropna()
                        if len(df) >= 2:
                            result[sym] = df
                    except Exception:
                        continue
            else:
                sym = batch[0]
                df  = raw.rename(columns=str.lower).dropna()
                if len(df) >= 2:
                    result[sym] = df

        except Exception as e:
            log.warning(f"  OHLCV batch error at {i}: {e}")
            continue

    log.info(f"  Got OHLCV for {len(result):,} symbols")
    return result


def extract_day(ohlcv, target_date):
    date_str = target_date.strftime("%Y-%m-%d")
    mask = ohlcv.index.strftime("%Y-%m-%d") == date_str
    if not mask.any():
        return {"open": None, "high": None, "low": None, "close": None}
    row = ohlcv[mask].iloc[0]
    return {k: float(row[k]) if pd.notna(row[k]) else None
            for k in ["open", "high", "low", "close"]}


def pct(entry, exit_):
    if entry and exit_ and entry > 0:
        return round((exit_ / entry) - 1, 6)
    return None


def compute_returns(prior_close, days):
    weekly_highs = [d["high"] for d in days.values() if d["high"] is not None]
    weekly_lows  = [d["low"]  for d in days.values() if d["low"]  is not None]
    weekly_high  = max(weekly_highs) if weekly_highs else None
    weekly_low   = min(weekly_lows)  if weekly_lows  else None

    peak_day = None
    for d in ["mon", "tue", "wed", "thu", "fri"]:
        if days[d]["high"] is not None and days[d]["high"] == weekly_high:
            peak_day = d
            break

    return {
        "return_fri_fri":  pct(prior_close,         days["fri"]["close"]),
        "return_mon_open": pct(days["mon"]["open"],  days["fri"]["close"]),
        "return_tue_open": pct(days["tue"]["open"],  days["fri"]["close"]),
        "return_wed_open": pct(days["wed"]["open"],  days["fri"]["close"]),
        "return_fri_mon":  pct(prior_close,          days["mon"]["close"]),
        "return_mon_tue":  pct(days["mon"]["close"], days["tue"]["open"]),
        "return_mon_peak": pct(days["mon"]["open"],  weekly_high),
        "return_tue_peak": pct(days["tue"]["open"],  weekly_high),
        "weekly_high":     weekly_high,
        "weekly_low":      weekly_low,
        "peak_day":        peak_day,
    }


def build_rows(scores, ohlcv_data, week_start_friday):
    dates = get_week_dates(week_start_friday)
    rows  = []

    for _, sr in scores.iterrows():
        sym   = sr["symbol"]
        ohlcv = ohlcv_data.get(sym)
        empty = {"open": None, "high": None, "low": None, "close": None}

        if ohlcv is None:
            prior_close = None
            day_prices  = {d: empty.copy() for d in ["mon","tue","wed","thu","fri"]}
        else:
            prior_mask  = ohlcv.index.strftime("%Y-%m-%d") == week_start_friday.strftime("%Y-%m-%d")
            prior_close = float(ohlcv[prior_mask]["close"].iloc[0]) if prior_mask.any() else None
            day_prices  = {d: extract_day(ohlcv, dates[d]) for d in ["mon","tue","wed","thu","fri"]}

        rets = compute_returns(prior_close, day_prices)

        rows.append({
            "week_of":              week_start_friday.strftime("%Y-%m-%d"),
            "symbol":               sym,
            "sector":               sr.get("sector"),
            "regime":               sr.get("regime"),
            "regime_composite":     sr.get("regime_composite"),
            "alpha_score":          sr.get("alpha_score"),
            "alpha_rank":           sr.get("alpha_rank"),
            "alpha_pct_rank":       sr.get("alpha_pct_rank"),
            "conviction":           sr.get("conviction"),
            "ev_score":             sr.get("ev_score"),
            "ev_rank":              sr.get("ev_rank"),
            "ev_pct_rank":          sr.get("ev_pct_rank"),
            "avg_win_magnitude":    sr.get("avg_win_magnitude"),
            "avg_loss_magnitude":   sr.get("avg_loss_magnitude"),
            "weekly_vol_predicted": sr.get("weekly_vol"),
            "composite_rank":       sr.get("composite_rank"),
            "sig_momentum":         sr.get("sig_momentum"),
            "sig_catalyst":         sr.get("sig_catalyst"),
            "sig_fundamentals":     sr.get("sig_fundamentals"),
            "sig_sentiment":        sr.get("sig_sentiment"),
            "sig_momentum_adj":     sr.get("sig_momentum_adj"),
            "sig_catalyst_adj":     sr.get("sig_catalyst_adj"),
            "sig_fundamentals_adj": sr.get("sig_fundamentals_adj"),
            "sig_sentiment_adj":    sr.get("sig_sentiment_adj"),
            "prior_close":          prior_close,
            "mon_open":             day_prices["mon"]["open"],
            "mon_close":            day_prices["mon"]["close"],
            "tue_open":             day_prices["tue"]["open"],
            "tue_close":            day_prices["tue"]["close"],
            "wed_open":             day_prices["wed"]["open"],
            "wed_close":            day_prices["wed"]["close"],
            "thu_open":             day_prices["thu"]["open"],
            "thu_close":            day_prices["thu"]["close"],
            "fri_open":             day_prices["fri"]["open"],
            "fri_close":            day_prices["fri"]["close"],
            "weekly_high":          rets.get("weekly_high"),
            "weekly_low":           rets.get("weekly_low"),
            "peak_day":             rets.get("peak_day"),
            "return_fri_fri":       rets.get("return_fri_fri"),
            "return_mon_open":      rets.get("return_mon_open"),
            "return_tue_open":      rets.get("return_tue_open"),
            "return_wed_open":      rets.get("return_wed_open"),
            "return_fri_mon":       rets.get("return_fri_mon"),
            "return_mon_tue":       rets.get("return_mon_tue"),
            "return_mon_peak":      rets.get("return_mon_peak"),
            "return_tue_peak":      rets.get("return_tue_peak"),
            "label":                np.nan,
        })

    df = pd.DataFrame(rows)

    # Primary return for model = Tuesday open to Friday close
    if df.get("return_tue_open") is not None and df["return_tue_open"].notna().any():
        df["forward_return_1w"]      = df["return_tue_open"]
    else:
        df["forward_return_1w"]      = df["return_fri_fri"]

    df["forward_return_1w_rank"] = df["forward_return_1w"].rank(pct=True).round(4)
    df["label"] = (df["forward_return_1w_rank"] >= 0.80).astype(float)

    return df


def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not assert_normal_week("collect_returns"):
        return

    today_ct  = now_ct().strftime("%Y-%m-%d")
    lock_file = DATA_DIR / ".collect_lock"
    lock_key  = f"friday_learning_{today_ct}"
    if lock_file.exists() and lock_file.read_text().strip() == lock_key:
        log.info(f"collect_returns already ran today ({today_ct}) — skipping duplicate")
        return
    lock_file.write_text(lock_key)

    if not SCORES.exists():
        log.warning("No scores_final.csv — nothing to collect")
        return

    scores = pd.read_csv(SCORES)
    log.info(f"Loaded scores: {len(scores):,} tickers")

    today        = datetime.today()
    this_friday  = last_friday(today)
    score_friday = this_friday - timedelta(weeks=1)

    # 5-trading-day guard
    scored_at = None
    if "scored_at" in scores.columns:
        try:
            scored_at = datetime.strptime(str(scores["scored_at"].iloc[0]), "%Y-%m-%d")
        except Exception:
            pass

    if scored_at:
        elapsed = trading_days_since(scored_at)
        log.info(f"Scores generated: {scored_at.date()}  |  Elapsed: {elapsed} trading days")
        if elapsed < 5:
            log.warning(f"Only {elapsed} trading days elapsed (need 5) — skipping")
            return

    # Idempotency check
    week_str = score_friday.strftime("%Y-%m-%d")
    if PERF_LOG.exists():
        existing = pd.read_csv(PERF_LOG, low_memory=False)
        if "week_of" in existing.columns and (existing["week_of"] == week_str).any():
            log.info(f"Week {week_str} already logged — skipping")
            return
    else:
        existing = pd.DataFrame()

    # Fetch OHLCV
    symbols    = scores["symbol"].tolist()
    ohlcv_data = fetch_weekly_ohlcv(symbols, score_friday)

    if not ohlcv_data:
        log.warning("No OHLCV data fetched")
        return

    # Build rows
    log.info("Building performance log rows...")
    new_df = build_rows(scores, ohlcv_data, score_friday)

    combined = pd.concat([existing, new_df], ignore_index=True) \
               if not existing.empty else new_df
    combined.to_csv(PERF_LOG, index=False)

    coverage = new_df["return_tue_open"].notna().mean()
    log.info(f"Performance log updated: {len(new_df):,} rows added")
    log.info(f"Return coverage: {coverage*100:.1f}%")

    # Entry day comparison for top 20% composite picks
    top20 = new_df.nsmallest(max(1, int(len(new_df)*0.20)), "composite_rank")
    log.info(f"\nEntry day comparison — top 20% composite rank ({len(top20)} tickers):")
    for col, label in [
        ("return_mon_open", "Monday open → Friday close"),
        ("return_tue_open", "Tuesday open → Friday close  <- primary"),
        ("return_mon_peak", "Monday open → weekly high"),
        ("return_tue_peak", "Tuesday open → weekly high"),
        ("return_fri_fri",  "Friday close → Friday close (theoretical)"),
    ]:
        if col in top20.columns:
            avg = top20[col].mean()
            if pd.notna(avg):
                log.info(f"  {label:<48} {avg*100:+.2f}%")

    if "peak_day" in new_df.columns:
        peak_dist = new_df["peak_day"].value_counts(normalize=True)
        log.info(f"\nWeekly high occurred on (full universe):")
        for day, pct_val in peak_dist.items():
            log.info(f"  {day}: {pct_val*100:.1f}%")

    log.info(f"\nOutput -> {PERF_LOG}  (total rows: {len(combined):,})")


if __name__ == "__main__":
    run()
