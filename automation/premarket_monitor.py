"""
automation/premarket_monitor.py
================================
Runs every Monday starting at 7:00 AM Eastern.
Checks pre-market prices for the top ranked tickers every 30 minutes
until market open (9:30 AM Eastern = 5 checks total).

Each check compares to all previous checks to determine:
    - Gap % from Friday close
    - Direction trend (ramping / stable / fading)
    - Velocity (acceleration between checks)

Writes:
    data/premarket_log.json          raw check data (all 5 checks)
    reports/YYYY-MM-DD_premarket.html  live-updating report

The report re-renders after each check so it's always current
when you open it. Open it once at 8:30 CT and refresh at 9:00.
"""

import json
import time
import sys
import logging
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from jinja2 import Template

# Ensure repo root is on path so automation modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")

DATA_DIR    = Path("data")
REPORTS     = Path("docs/reports")
SCORES_CSV  = DATA_DIR / "scores_final.csv"
PRELOG      = DATA_DIR / "premarket_log.json"

# Check schedule: minutes after 7:00 AM Eastern
CHECK_TIMES_MINUTES = [0, 30, 60, 90, 120]   # 7:00, 7:30, 8:00, 8:30, 9:00 AM ET
TOP_N               = 30    # watch top N tickers from Friday's scores
GAP_WARN_PCT        = 0.10  # flag if pre-market gap > 10%
GAP_SKIP_PCT        = 0.25  # strong skip signal if gap > 25%


# ── PRICE FETCH ───────────────────────────────────────────────────────────────

def fetch_premarket_prices(symbols: list[str]) -> dict[str, dict]:
    """
    Fetch current pre-market price and most recent prior session close.
    Works for both Monday (uses Friday close) and Tuesday (uses Monday close).
    Returns {symbol: {premarket_price, prior_close, gap_pct, volume}}
    """
    results = {}
    end     = datetime.date.today()
    start   = end - datetime.timedelta(days=7)  # enough to catch Mon or Fri close

    # Get prior session closes via regular history (reliable)
    try:
        closes_raw = yf.download(
            symbols,
            start=str(start),
            end=str(end),
            auto_adjust=True,
            progress=False,
            threads=True,
        )["Close"]
        prior_closes = {}
        for sym in symbols:
            try:
                col = closes_raw[sym] if len(symbols) > 1 else closes_raw
                prior_closes[sym] = float(col.dropna().iloc[-1])
            except Exception:
                continue
    except Exception as e:
        log.warning(f"  Prior close batch failed: {e}")
        prior_closes = {}

    # Get pre-market prices via 1m data with prepost=True
    batch_size = 10
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            pm_raw = yf.download(
                batch,
                period="1d",
                interval="1m",
                prepost=True,
                auto_adjust=True,
                progress=False,
                threads=True,
            )["Close"]

            for sym in batch:
                try:
                    prior_close = prior_closes.get(sym)
                    if prior_close is None:
                        continue
                    col = pm_raw[sym] if len(batch) > 1 else pm_raw
                    col = col.dropna()
                    if col.empty:
                        continue
                    premarket = float(col.iloc[-1])
                    gap_pct   = (premarket / prior_close) - 1
                    results[sym] = {
                        "premarket_price": round(premarket, 2),
                        "prior_close":     round(prior_close, 2),
                        "gap_pct":         round(gap_pct, 4),
                        "pm_volume":       0,
                    }
                except Exception as e:
                    log.debug(f"  {sym}: {e}")
                    continue

        except Exception as e:
            log.warning(f"  Pre-market batch {i//batch_size+1} failed: {e}")
            for sym in batch:
                try:
                    prior_close = prior_closes.get(sym)
                    if prior_close is None:
                        continue
                    fi = yf.Ticker(sym).fast_info
                    premarket = getattr(fi, "pre_market_price", None) or \
                                getattr(fi, "last_price", None)
                    if premarket is None:
                        continue
                    results[sym] = {
                        "premarket_price": round(float(premarket), 2),
                        "prior_close":     round(prior_close, 2),
                        "gap_pct":         round((float(premarket)/prior_close)-1, 4),
                        "pm_volume":       0,
                    }
                except Exception:
                    continue

    return results


# ── TREND ANALYSIS ────────────────────────────────────────────────────────────

def analyze_trend(symbol: str, checks: list[dict]) -> dict:
    """
    Given all checks so far for a symbol, determine trend direction
    and velocity.

    checks: list of {time, gap_pct, premarket_price} ordered oldest first
    """
    if len(checks) < 2:
        return {"trend": "initial", "velocity": 0.0, "label": "—", "arrow": "·", "sparkline": "—"}

    gaps = [c["gap_pct"] for c in checks]

    # Velocity: average change per interval
    deltas   = [gaps[i] - gaps[i-1] for i in range(1, len(gaps))]
    velocity = float(np.mean(deltas))

    # Trend classification
    recent_delta = gaps[-1] - gaps[-2]

    if len(gaps) >= 3:
        # Linear regression slope over all checks
        x     = np.arange(len(gaps))
        slope = float(np.polyfit(x, gaps, 1)[0])
    else:
        slope = recent_delta

    if slope > 0.005:
        trend = "ramping"
        arrow = "↑↑" if slope > 0.015 else "↑"
    elif slope < -0.005:
        trend = "fading"
        arrow = "↓↓" if slope < -0.015 else "↓"
    else:
        trend = "stable"
        arrow = "→"

    # Build sparkline string  e.g. "+31% +38% +44% +47%"
    sparkline = "  ".join(
        f"{g*100:+.1f}%" for g in gaps
    )

    return {
        "trend":     trend,
        "velocity":  round(velocity, 4),
        "slope":     round(slope, 4),
        "arrow":     arrow,
        "sparkline": sparkline,
        "label":     f"{trend} ({arrow})",
    }


def action_recommendation(gap_pct: float, trend: str, alpha_rank: int) -> dict:
    """
    Plain-English action guidance based on gap + trend.
    """
    g = gap_pct

    if g > GAP_SKIP_PCT and trend == "ramping":
        return {
            "action":  "WATCH",
            "color":   "amber",
            "reason":  f"Gap >{GAP_SKIP_PCT*100:.0f}% and still ramping — possible squeeze. Monitor closely, wait for pullback or confirmation candle.",
        }
    elif g > GAP_SKIP_PCT and trend == "fading":
        return {
            "action":  "SKIP",
            "color":   "red",
            "reason":  f"Gapped >{GAP_SKIP_PCT*100:.0f}% and fading — move likely exhausted. Skip, revisit next week.",
        }
    elif g > GAP_SKIP_PCT:
        return {
            "action":  "SKIP",
            "color":   "red",
            "reason":  f"Gap >{GAP_SKIP_PCT*100:.0f}% — risk/reward degraded from Friday score. Skip unless you have strong conviction.",
        }
    elif g > GAP_WARN_PCT and trend == "ramping":
        return {
            "action":  "CAUTION",
            "color":   "amber",
            "reason":  f"Gap {g*100:.1f}% and building — enter smaller than planned or wait for open.",
        }
    elif g > GAP_WARN_PCT and trend == "fading":
        return {
            "action":  "WAIT",
            "color":   "amber",
            "reason":  f"Gap {g*100:.1f}% but fading — may give a better entry at open. Watch first 5 min.",
        }
    elif g > GAP_WARN_PCT:
        return {
            "action":  "CAUTION",
            "color":   "amber",
            "reason":  f"Gap {g*100:.1f}% — proceed carefully. Risk/reward slightly reduced.",
        }
    elif g < -0.05 and trend == "fading":
        return {
            "action":  "REVIEW",
            "color":   "red",
            "reason":  f"Gapping down {g*100:.1f}% — check for negative news before entering.",
        }
    else:
        return {
            "action":  "GO",
            "color":   "green",
            "reason":  f"Gap {g*100:+.1f}% — thesis intact, proceed as planned.",
        }


# ── HTML REPORT ───────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="300">
<title>PRE-MARKET // {{ day_name }} {{ date }}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#080c0f; --bg2:#0d1318; --bg3:#111820;
  --border:#1e2d38; --border2:#243545;
  --text:#c8d8e4; --text2:#6a8a9f; --text3:#3d5a6b;
  --accent:#00c8ff; --green:#00e676; --amber:#ffab00; --red:#ff4444;
  --mono:'IBM Plex Mono',monospace;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:var(--mono);font-size:13px;line-height:1.5;}
.header{padding:14px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}
.logo{font-size:18px;font-weight:600;letter-spacing:.15em;color:var(--accent);}
.logo span{color:var(--text3);font-weight:300;}
.meta{font-size:11px;color:var(--text2);}
.checks-strip{display:flex;gap:0;border-bottom:1px solid var(--border);}
.check-pill{padding:8px 20px;font-size:10px;letter-spacing:.08em;border-right:1px solid var(--border);color:var(--text3);}
.check-pill.done{color:var(--text2);}
.check-pill.active{color:var(--accent);border-bottom:2px solid var(--accent);margin-bottom:-1px;}
.check-pill.pending{color:var(--text3);}
table{width:100%;border-collapse:collapse;font-size:12px;}
thead th{background:var(--bg2);padding:8px 12px;text-align:left;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--text2);border-bottom:1px solid var(--border2);white-space:nowrap;}
tbody tr{border-bottom:1px solid var(--border);}
tbody tr:hover{background:var(--bg2);}
td{padding:7px 12px;white-space:nowrap;vertical-align:middle;}
.sym{font-weight:600;font-size:13px;letter-spacing:.04em;}
.gap-pos{color:var(--green);}
.gap-neg{color:var(--red);}
.gap-warn{color:var(--amber);}
.spark{color:var(--text2);font-size:10px;letter-spacing:.02em;}
.trend-ramping{color:var(--green);}
.trend-fading{color:var(--red);}
.trend-stable{color:var(--text2);}
.trend-initial{color:var(--text3);}
.action{display:inline-block;padding:2px 8px;font-size:10px;font-weight:600;letter-spacing:.08em;border-radius:2px;}
.action-GO{background:rgba(0,230,118,.12);color:var(--green);border:1px solid rgba(0,230,118,.3);}
.action-CAUTION{background:rgba(255,171,0,.10);color:var(--amber);border:1px solid rgba(255,171,0,.3);}
.action-WATCH{background:rgba(255,171,0,.10);color:var(--amber);border:1px solid rgba(255,171,0,.3);}
.action-WAIT{background:rgba(255,171,0,.08);color:var(--amber);border:1px solid rgba(255,171,0,.2);}
.action-SKIP{background:rgba(255,68,68,.12);color:var(--red);border:1px solid rgba(255,68,68,.3);}
.action-REVIEW{background:rgba(255,68,68,.10);color:var(--red);border:1px solid rgba(255,68,68,.25);}
.reason{color:var(--text2);font-size:10px;max-width:320px;white-space:normal;line-height:1.4;}
.rank{color:var(--text3);font-size:11px;}
.thesis{color:var(--text3);font-size:10px;max-width:200px;white-space:normal;line-height:1.4;}
.footer{padding:10px 24px;border-top:1px solid var(--border);font-size:10px;color:var(--text3);display:flex;justify-content:space-between;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px;}

/* ── DECISION GUIDE ── */
.guide-bar{
  padding:8px 24px;
  background:var(--bg2);
  border-bottom:1px solid var(--border);
  display:flex;
  align-items:center;
  gap:12px;
  cursor:pointer;
  user-select:none;
}
.guide-bar:hover{background:var(--bg3);}
.guide-label{font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--text3);}
.guide-pills{display:flex;gap:6px;flex-wrap:wrap;}
.gpill{padding:2px 8px;font-size:9px;font-weight:600;letter-spacing:.08em;border-radius:2px;cursor:pointer;}
.gpill-go    {background:rgba(0,230,118,.12);color:var(--green);border:1px solid rgba(0,230,118,.3);}
.gpill-caution{background:rgba(255,171,0,.10);color:var(--amber);border:1px solid rgba(255,171,0,.3);}
.gpill-skip  {background:rgba(255,68,68,.12);color:var(--red);border:1px solid rgba(255,68,68,.3);}
.gpill-wait  {background:rgba(255,171,0,.08);color:var(--amber);border:1px solid rgba(255,171,0,.2);}
.guide-toggle{margin-left:auto;font-size:10px;color:var(--text3);}

.guide-panel{
  display:none;
  background:var(--bg3);
  border-bottom:2px solid var(--border2);
  padding:16px 24px 20px;
}
.guide-panel.open{display:grid;grid-template-columns:1fr 1fr;gap:20px;}
@media(max-width:600px){.guide-panel.open{grid-template-columns:1fr;}}

.guide-section-title{
  font-size:10px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--text3);margin-bottom:10px;padding-bottom:6px;
  border-bottom:1px solid var(--border);
}
.guide-step{
  display:flex;gap:10px;margin-bottom:8px;font-size:11px;line-height:1.5;
}
.guide-step-num{
  flex-shrink:0;width:18px;height:18px;border-radius:50%;
  background:var(--bg2);border:1px solid var(--border2);
  display:flex;align-items:center;justify-content:center;
  font-size:9px;color:var(--text3);margin-top:1px;
}
.guide-step-body{color:var(--text2);}
.guide-step-body strong{color:var(--text);font-weight:500;}
.guide-rule{
  font-size:10px;line-height:1.6;color:var(--text2);
  padding:6px 10px;background:var(--bg2);border-radius:4px;
  border-left:2px solid var(--border2);margin-bottom:6px;
}
.guide-rule.r-go    {border-left-color:var(--green);}
.guide-rule.r-caution{border-left-color:var(--amber);}
.guide-rule.r-skip  {border-left-color:var(--red);}
.guide-rule .rl{font-weight:500;}
.guide-rule .rl.go    {color:var(--green);}
.guide-rule .rl.caution{color:var(--amber);}
.guide-rule .rl.skip  {color:var(--red);}
</style>
</head>
<body>
<div class="header">
  <div style="display:flex;align-items:center;gap:16px">
    <a href="../index.html" style="font-size:10px;color:var(--text3);text-decoration:none;letter-spacing:.08em;border:1px solid var(--border);padding:4px 10px;border-radius:2px">← HUB</a>
    <div>
      <div class="logo">PRE-MARKET<span>//</span>{{ day_name }}</div>
      <div class="meta">{{ date }} &nbsp;·&nbsp; top {{ n_tickers }} tickers &nbsp;·&nbsp; auto-refreshes every 5 min</div>
    </div>
  </div>
  <div class="meta" style="text-align:right">
    last check: <span style="color:var(--accent)">{{ last_check_time }}</span><br>
    next check: <span style="color:var(--text2)">{{ next_check_time }}</span>
  </div>
</div>

<div class="checks-strip">
{% for c in check_labels %}
  <div class="check-pill {{ c.status }}">{{ c.label }}</div>
{% endfor %}
</div>

<!-- DECISION GUIDE -->
<div class="guide-bar" onclick="toggleGuide()">
  <span class="guide-label">Entry decision guide</span>
  <div class="guide-pills">
    <span class="gpill gpill-go">GO = enter</span>
    <span class="gpill gpill-caution">CAUTION = smaller / wait</span>
    <span class="gpill gpill-wait">WAIT = Tuesday</span>
    <span class="gpill gpill-skip">SKIP = pass entirely</span>
  </div>
  <span class="guide-toggle" id="guide-toggle-label">▼ expand</span>
</div>

<div class="guide-panel" id="guide-panel">

  <div>
    <div class="guide-section-title">Step-by-step entry procedure</div>

    <div class="guide-step">
      <div class="guide-step-num">1</div>
      <div class="guide-step-body">
        <strong>Check regime (index page).</strong><br>
        Risk-on → bias Monday. Everything else → bias Tuesday unless overridden below.
      </div>
    </div>

    <div class="guide-step">
      <div class="guide-step-num">2</div>
      <div class="guide-step-body">
        <strong>Open pre-market monitor at 7:30am CT.</strong><br>
        Read the Action column. It is binary.</div>
    </div>

    <div class="guide-step">
      <div class="guide-step-num">3</div>
      <div class="guide-step-body">
        <strong>Read Trend + Gap together.</strong> See rules →
      </div>
    </div>

    <div class="guide-step">
      <div class="guide-step-num">4</div>
      <div class="guide-step-body">
        <strong>Confirm with sparkline.</strong><br>
        Last 2 checks both higher than prior 2 → ramping confirmed.<br>
        Last 2 checks both lower → fading confirmed.<br>
        Alternating → stable, use gap size rule.
      </div>
    </div>

    <div class="guide-step">
      <div class="guide-step-num">5</div>
      <div class="guide-step-body">
        <strong>If deferred to Tuesday:</strong> check Monday close vs Friday close.<br>
        Monday closed above Friday → enter Tuesday open.<br>
        Monday closed &gt;5% below Friday → check for news, skip if none found.<br>
        Monday closed within 5% either direction → enter Tuesday open.
      </div>
    </div>
  </div>

  <div>
    <div class="guide-section-title">Trend + gap decision rules</div>

    <div class="guide-rule r-go">
      <span class="rl go">GO — Monday open</span><br>
      Gap &lt;10% + any trend · Gap &lt;25% + ramping (↑↑)
    </div>

    <div class="guide-rule r-go">
      <span class="rl go">GO — Monday open</span><br>
      Gap &lt;10% + stable (→)
    </div>

    <div class="guide-rule r-caution">
      <span class="rl caution">TUESDAY — defer entry</span><br>
      Gap 10-25% + stable (→) · Gap any + fading (↓) over 10%
    </div>

    <div class="guide-rule r-caution">
      <span class="rl caution">CAUTION — enter smaller</span><br>
      Gap 10-25% + ramping · reduce position size, not thesis
    </div>

    <div class="guide-rule r-skip">
      <span class="rl skip">SKIP — pass entirely</span><br>
      Gap &gt;25% + fading (↓↓) · move exhausted, revisit next week
    </div>

    <div class="guide-rule r-skip">
      <span class="rl skip">REVIEW — check news first</span><br>
      Gapping down &gt;5% + fading · specific neg catalyst = skip,
      no news = treat as caution
    </div>

    <div class="guide-section-title" style="margin-top:14px">WATCH action</div>
    <div class="guide-rule r-caution">
      <span class="rl caution">WATCH = possible squeeze</span><br>
      Gap &gt;25% still ramping. Do not chase at open.
      Wait for first 5-min candle to close, enter on pullback only.
      Set hard stop at pre-market low.
    </div>
  </div>

</div>

<table>
<thead>
  <tr>
    <th>Rank</th>
    <th>Symbol</th>
    <th>Sector</th>
    <th>Prior Close</th>
    <th>Pre-Mkt</th>
    <th>Gap</th>
    <th>Trend</th>
    <th>Checks</th>
    <th>Action</th>
    <th>Reason</th>
    <th>Thesis</th>
  </tr>
</thead>
<tbody>
{% for row in rows %}
<tr>
  <td class="rank">#{{ row.rank }}</td>
  <td class="sym">{{ row.symbol }}</td>
  <td style="color:var(--text3);font-size:10px">{{ row.sector }}</td>
  <td style="text-align:right">${{ row.prior_close }}</td>
  <td style="text-align:right;color:var(--accent)">${{ row.premarket_price }}</td>
  <td class="{% if row.gap_pct > 0.10 %}gap-warn{% elif row.gap_pct > 0 %}gap-pos{% else %}gap-neg{% endif %}" style="text-align:right;font-weight:500">
    {{ row.gap_str }}
  </td>
  <td class="trend-{{ row.trend }}">{{ row.arrow }} {{ row.trend }}</td>
  <td class="spark">{{ row.sparkline }}</td>
  <td><span class="action action-{{ row.action }}">{{ row.action }}</span></td>
  <td class="reason">{{ row.reason }}</td>
  <td class="thesis">{{ row.thesis }}</td>
</tr>
{% endfor %}
</tbody>
</table>

<div class="footer">
  <span>MOMENTUM//ALPHA Pre-Market Monitor · {{ date }}</span>
  <span>Prices delayed · Not investment advice · Always verify with your broker</span>
</div>
<script>
function toggleGuide() {
  const panel  = document.getElementById('guide-panel');
  const label  = document.getElementById('guide-toggle-label');
  const isOpen = panel.classList.toggle('open');
  label.textContent = isOpen ? '▲ collapse' : '▼ expand';
}
</script>
</body>
</html>"""


def render_report(
    rows: list[dict],
    checks_done: int,
    date_str: str,
    last_check_time: str,
    day_name: str = "MONDAY",
) -> str:
    check_names = ["6:00 AM CT", "6:30 AM CT", "7:00 AM CT", "7:30 AM CT", "8:00 AM CT"]
    check_labels = []
    for i, name in enumerate(check_names):
        if i < checks_done:
            status = "done"
        elif i == checks_done:
            status = "active"
        else:
            status = "pending"
        check_labels.append({"label": name, "status": status})

    next_idx = min(checks_done, len(check_names) - 1)
    next_time = check_names[next_idx] if checks_done < len(check_names) else "complete"

    tmpl = Template(HTML)
    return tmpl.render(
        date=date_str,
        day_name=day_name,
        n_tickers=len(rows),
        last_check_time=last_check_time,
        next_check_time=next_time,
        check_labels=check_labels,
        rows=rows,
    )


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Holiday check -- skip if not a full 5-day trading week
    from automation.tz_utils import assert_normal_week, now_ct
    if not assert_normal_week("premarket_monitor"):
        return

    # DST guard -- both CDT and CST crons are scheduled, only run at 6:00 AM CT
    # Whichever fires at the wrong hour exits immediately to prevent double-running
    _now_ct = now_ct()
    if _now_ct.hour != 6:
        log.info(f"DST guard: current CT hour is {_now_ct.hour}, expected 6 -- skipping")
        return

    date_str   = datetime.date.today().strftime("%Y-%m-%d")
    day_name   = datetime.date.today().strftime("%A").upper()  # MONDAY or TUESDAY
    report_out = REPORTS / f"{date_str}_premarket.html"

    # Load Friday's top tickers
    if not SCORES_CSV.exists():
        log.error("No scores_final.csv found — run weekly pipeline first")
        return

    scores = pd.read_csv(SCORES_CSV)
    top    = scores.head(TOP_N).copy()
    symbols = top["symbol"].tolist()

    log.info(f"Monitoring {len(symbols)} tickers: {', '.join(symbols[:10])}...")

    # Load or init the pre-market log
    if PRELOG.exists():
        with open(PRELOG) as f:
            pm_log = json.load(f)
        # If it's a new day, reset
        if pm_log.get("date") != date_str:
            pm_log = {"date": date_str, "checks": []}
    else:
        pm_log = {"date": date_str, "checks": []}

    checks_done = len(pm_log["checks"])
    remaining   = CHECK_TIMES_MINUTES[checks_done:]

    if not remaining:
        log.info("All checks already complete for today")
        return

    for check_idx, wait_minutes in enumerate(remaining):
        absolute_check = checks_done + check_idx
        check_name     = ["6:00","6:30","7:00","7:30","8:00"][absolute_check]

        # Sleep until this check window
        if check_idx > 0:
            sleep_secs = 30 * 60   # 30 minutes
            log.info(f"Sleeping {sleep_secs//60} min until {check_name} AM check...")
            time.sleep(sleep_secs)

        log.info(f"Running check {absolute_check + 1}/5 ({check_name} AM ET)...")
        now_str = datetime.datetime.now().strftime("%H:%M:%S")

        prices = fetch_premarket_prices(symbols)
        log.info(f"  Got prices for {len(prices)} symbols")

        pm_log["checks"].append({
            "check_num":  absolute_check + 1,
            "check_name": check_name,
            "timestamp":  now_str,
            "prices":     prices,
        })

        with open(PRELOG, "w") as f:
            json.dump(pm_log, f, indent=2)

        # Build rows for report
        rows = []
        for _, ticker_row in top.iterrows():
            sym = ticker_row["symbol"]

            # Gather all checks for this symbol
            sym_checks = []
            for chk in pm_log["checks"]:
                if sym in chk["prices"]:
                    sym_checks.append({
                        "time":           chk["check_name"],
                        "gap_pct":        chk["prices"][sym]["gap_pct"],
                        "premarket_price": chk["prices"][sym]["premarket_price"],
                    })

            latest = prices.get(sym)
            if not latest:
                continue

            trend_data = analyze_trend(sym, sym_checks)
            action     = action_recommendation(
                latest["gap_pct"],
                trend_data["trend"],
                int(ticker_row.get("alpha_rank", 0)),
            )

            gap_pct = latest["gap_pct"]
            rows.append({
                "rank":           int(ticker_row.get("alpha_rank", 0)),
                "symbol":         sym,
                "sector":         str(ticker_row.get("sector", ""))[:20],
                "prior_close":   latest["prior_close"],
                "premarket_price": latest["premarket_price"],
                "gap_pct":        gap_pct,
                "gap_str":        f"{gap_pct*100:+.1f}%",
                "trend":          trend_data["trend"],
                "arrow":          trend_data["arrow"],
                "sparkline":      trend_data["sparkline"],
                "action":         action["action"],
                "reason":         action["reason"],
                "thesis":         str(ticker_row.get("thesis", ""))[:120],
            })

        # Sort: SKIPs to bottom, then by gap desc
        action_order = {"SKIP":5,"REVIEW":4,"WATCH":1,"CAUTION":2,"WAIT":3,"GO":0}
        rows.sort(key=lambda r: (action_order.get(r["action"],0), -abs(r["gap_pct"])))

        html = render_report(rows, absolute_check + 1, date_str, now_str, day_name)
        with open(report_out, "w", encoding="utf-8") as f:
            f.write(html)

        log.info(f"  Report updated -> {report_out}")

        # Update hub index so premarket link is accessible from main page
        try:
            import importlib
            idx = importlib.import_module("automation.update_index")
            idx.run()
        except Exception as e:
            log.warning(f"  Index update failed: {e}")

        # Commit after each check so the page updates in real time
        try:
            import subprocess
            subprocess.run(["git", "config", "user.name", "momentum-bot"], check=False)
            subprocess.run(["git", "config", "user.email", "momentum-bot@users.noreply.github.com"], check=False)
            subprocess.run(["git", "add", "docs/", "data/premarket_log.json"], check=False)
            result = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                capture_output=True
            )
            if result.returncode != 0:
                subprocess.run(
                    ["git", "commit", "-m",
                     f"premarket: {date_str} check {absolute_check + 1}/5 ({check_name} AM)"],
                    check=False
                )
                subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=False)
                subprocess.run(["git", "push"], check=False)
                log.info(f"  Committed check {absolute_check + 1}/5")
        except Exception as e:
            log.warning(f"  Git commit failed: {e}")

        # Log summary
        skips   = sum(1 for r in rows if r["action"] == "SKIP")
        caution = sum(1 for r in rows if r["action"] in ("CAUTION","WATCH","WAIT"))
        go      = sum(1 for r in rows if r["action"] == "GO")
        log.info(f"  GO: {go}  CAUTION: {caution}  SKIP: {skips}")

        for r in rows[:5]:
            log.info(f"  #{r['rank']:<4} {r['symbol']:<8} {r['gap_str']:>7}  "
                     f"{r['arrow']} {r['trend']:<8}  [{r['action']}]")

    log.info("Pre-market monitor complete")


if __name__ == "__main__":
    run()
