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
from automation.system_logger import log_event, LogStatus
from automation.notifier import notify_alert, notify_error

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

    # Sub-signals needed for retrain.py to train the full 18-feature model.
    # Previously only the 4 coarse composites + 4 _adj composites were pulled;
    # retrain.py lists 18 RETRAIN_FEATURES but build_training_data() only used
    # the 4 coarse ones, producing a feature-shape mismatch with stage 4's
    # model. Now we carry every sub-signal through. Missing columns in scores
    # (e.g. from pre-fix runs) resolve to None via sr.get(), which retrain
    # will then drop via the notna() mask.
    SUB_SIGNAL_COLS = [
        "sig_momentum_rs", "sig_momentum_trend", "sig_momentum_vol_surge",
        "sig_momentum_breakout",
        "sig_catalyst_earnings", "sig_catalyst_insider", "sig_catalyst_analyst",
        "sig_fund_growth", "sig_fund_quality", "sig_fund_profitability",
        "sig_fund_value",
        "sig_sentiment_news", "sig_sentiment_analyst", "sig_sentiment_short",
    ]

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

        row = {
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
        }

        # Preserve the sub-signals that fed the model's prediction for this
        # week. These are what retrain.py feeds back in with real forward
        # returns as labels. sr.get() returns None if the column is missing,
        # which is the correct behavior for pre-fix rows (retrain drops them
        # via the notna mask).
        for col in SUB_SIGNAL_COLS:
            row[col] = sr.get(col)

        rows.append(row)

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
    log_event("collect_returns", LogStatus.INFO, "Starting collect_returns")

    if not assert_normal_week("collect_returns"):
        log_event("collect_returns", LogStatus.INFO,
                  "Skipped: not a normal trading week (disruptive holiday)")
        return

    today_ct  = now_ct().strftime("%Y-%m-%d")
    lock_file = DATA_DIR / ".collect_lock"
    lock_key  = f"friday_learning_{today_ct}"
    # Lock check: detects a successful prior run today. Write is moved to END of
    # function (was at top, but that blocked retries after any crash between lock
    # write and actual work completion).
    if lock_file.exists() and lock_file.read_text().strip() == lock_key:
        log.info(f"collect_returns already ran today ({today_ct}) — skipping duplicate")
        log_event("collect_returns", LogStatus.INFO,
                  f"Skipped: already ran today ({today_ct})")
        return

    if not SCORES.exists():
        log.warning("No scores_final.csv — nothing to collect")
        log_event("collect_returns", LogStatus.WARNING,
                  "Skipped: scores_final.csv missing")
        notify_alert("collect_returns",
                     "scores_final.csv missing — no data to collect. "
                     "Run the pipeline first.")
        return

    try:
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
                # This was THE silent failure mode that broke the learning loop. Prior
                # to the Tier 1 fix, weekend_refresh overwrote scored_at to the refresh
                # date, making elapsed always <5 and causing this guard to skip weekly.
                # Notification ensures we catch any recurrence immediately.
                log_event("collect_returns", LogStatus.WARNING,
                          f"Skipped: only {elapsed} trading days elapsed (need 5)",
                          metrics={"scored_at": str(scored_at.date()),
                                   "elapsed_trading_days": elapsed})
                notify_alert("collect_returns",
                             f"5-day guard skipped learning loop: "
                             f"scored_at={scored_at.date()}, elapsed={elapsed} days")
                return

        # Idempotency check
        week_str = score_friday.strftime("%Y-%m-%d")
        if PERF_LOG.exists():
            existing = pd.read_csv(PERF_LOG, low_memory=False)
            if "week_of" in existing.columns and (existing["week_of"] == week_str).any():
                log.info(f"Week {week_str} already logged — skipping")
                log_event("collect_returns", LogStatus.INFO,
                          f"Skipped: week {week_str} already in perf_log")
                return
        else:
            existing = pd.DataFrame()

        # Fetch OHLCV
        symbols    = scores["symbol"].tolist()
        ohlcv_data = fetch_weekly_ohlcv(symbols, score_friday)

        if not ohlcv_data:
            log.warning("No OHLCV data fetched")
            log_event("collect_returns", LogStatus.ERROR,
                      "Failed: yfinance returned no OHLCV data",
                      metrics={"symbols_requested": len(symbols)})
            notify_error("collect_returns",
                         f"yfinance returned no OHLCV data for "
                         f"{len(symbols)} symbols. Rate limit or network?")
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
        # Coerce composite_rank to numeric -- when the column mixes NaN from old
        # pre-Tier-2 rows with ints from new rows, pd.read_csv returns object
        # dtype and nsmallest can't sort it. `errors="coerce"` turns unparseable
        # values into NaN; dropna filters rows that couldn't be ranked.
        new_df["composite_rank"] = pd.to_numeric(
            new_df["composite_rank"], errors="coerce")
        ranked = new_df.dropna(subset=["composite_rank"])
        if len(ranked) > 0:
            top20 = ranked.nsmallest(max(1, int(len(ranked)*0.20)), "composite_rank")
        else:
            log.warning("  No rows with numeric composite_rank -- skipping entry day comparison")
            top20 = new_df.head(0)  # empty, so the loop below is a no-op
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

        # Success -- record lock AFTER work completes so partial failures don't
        # block retries within the same day.
        lock_file.write_text(lock_key)

        log_event("collect_returns", LogStatus.SUCCESS,
                  f"Collected {len(new_df):,} rows for week {week_str}",
                  metrics={
                      "week_of":          week_str,
                      "rows_added":       int(len(new_df)),
                      "total_rows":       int(len(combined)),
                      "coverage_pct":     round(float(coverage * 100), 1),
                      "symbols_with_data": int(len(ohlcv_data)),
                  })

    except Exception as e:
        log.error(f"collect_returns crashed: {e}", exc_info=True)
        log_event("collect_returns", LogStatus.ERROR,
                  "Unhandled exception during collect",
                  errors=[str(e)])
        notify_error("collect_returns", f"Unhandled error: {e}")
        raise


if __name__ == "__main__":
    run()
