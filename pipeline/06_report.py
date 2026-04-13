"""
Stage 6 — Report Generator
============================
Reads scores_final.csv + regime.json and renders a self-contained
HTML report. No external dependencies at report-open time —
everything is inlined.

The report is a single HTML file with embedded CSS + JS:
    - Regime banner (color-coded by state)
    - KPI summary strip (universe size, top conviction count, etc.)
    - Signal sparkbars per ticker
    - Full sortable, filterable, searchable ranked table (all tickers)
    - Expandable row detail: full signal breakdown + thesis
    - Sector filter chips
    - Conviction tier filter
    - Column sort on any header

Reads:   data/scores_final.csv, data/regime.json
Writes:  reports/YYYY-MM-DD.html
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from jinja2 import Template

log = logging.getLogger(__name__)

DATA_DIR   = Path("data")
REPORTS    = Path("docs/reports")
FINAL_CSV  = DATA_DIR / "scores_final.csv"
REGIME_JSON = DATA_DIR / "regime.json"


# ── DATA PREP ─────────────────────────────────────────────────────────────────

def load_and_prep() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(FINAL_CSV)
    with open(REGIME_JSON) as f:
        regime = json.load(f)

    # Clean up types
    numeric_cols = [c for c in df.columns if c.startswith("sig_") or
                    c in ["alpha_score", "alpha_pct_rank", "last_price",
                          "avg_vol_20d", "market_cap"]]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["conviction"]   = df["conviction"].fillna("low").astype(str)
    df["sector"]       = df["sector"].fillna("Unknown").astype(str)
    df["thesis"]       = df["thesis"].fillna("").astype(str)
    df["risk_flag"]    = df["risk_flag"].fillna("").astype(str)
    df["thesis_source"] = df.get("thesis_source", pd.Series(["rule_based"]*len(df))).fillna("rule_based").astype(str)

    # Format display columns
    df["price_fmt"]   = df["last_price"].apply(
        lambda x: f"${x:.2f}" if pd.notna(x) else "—")
    df["mcap_fmt"]    = df["market_cap"].apply(
        lambda x: f"${x/1e9:.1f}B" if pd.notna(x) and x >= 1e9
                  else (f"${x/1e6:.0f}M" if pd.notna(x) else "—"))
    df["vol_fmt"]     = df["avg_vol_20d"].apply(
        lambda x: f"{x/1e6:.1f}M" if pd.notna(x) and x >= 1e6
                  else (f"{x/1e3:.0f}K" if pd.notna(x) else "—"))
    df["score_fmt"]   = df["alpha_score"].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    df["rank_fmt"]    = df["alpha_rank"].apply(
        lambda x: f"#{int(x)}" if pd.notna(x) else "—")
    df["pct_fmt"]     = df["alpha_pct_rank"].apply(
        lambda x: f"{x*100:.1f}" if pd.notna(x) else "—")

    def sig_bar(val, width=40):
        """Convert 0-1 signal value to a percentage for CSS width."""
        if pd.isna(val):
            return 0
        return int(float(val) * width)

    # Signal bar widths (0-40 for CSS)
    for sig in ["sig_momentum", "sig_catalyst", "sig_fundamentals", "sig_sentiment"]:
        if sig in df.columns:
            df[f"{sig}_bar"] = df[sig].apply(sig_bar)

    return df, regime


def build_rows_json(df: pd.DataFrame) -> str:
    """Serialize all rows to JSON for the JS table engine."""
    rows = []
    for _, r in df.iterrows():
        def g(col, default=""):
            v = r.get(col, default)
            if pd.isna(v) if not isinstance(v, str) else False:
                return default
            return v

        def gf(col, decimals=4):
            v = r.get(col, None)
            try:
                return round(float(v), decimals) if pd.notna(v) else None
            except Exception:
                return None

        def suggested_stops(avg_win, weekly_vol):
            """
            Compute suggested stop parameters from signal data.
            hard_stop  = based on weekly vol tier (wider for volatile names)
            activation = 0.75 × avg_win  (switch to trail in winning territory)
            trail      = 0.50 × weekly_vol, capped at 15%
            """
            if not avg_win or not weekly_vol or avg_win <= 0 or weekly_vol <= 0:
                return None, None, None
            # Hard stop flexes with volatility
            if weekly_vol < 10:
                hard_stop = 7.0
            elif weekly_vol < 20:
                hard_stop = 10.0
            else:
                hard_stop = round(min(weekly_vol * 0.50, 15.0), 1)
            activation = round(avg_win * 0.75, 1)
            trail      = round(min(weekly_vol * 0.50, 15.0), 1)
            return hard_stop, activation, trail

        rows.append({
            "rank":       int(g("composite_rank", g("alpha_rank", 9999))),
            "alpha_rank": int(g("alpha_rank", 9999)),
            "ev_rank":    int(g("ev_rank", 9999)) if g("ev_rank", "") != "" else 9999,
            "symbol":     str(g("symbol")),
            "name":       str(g("name", "")),
            "sector":     str(g("sector", "Unknown")),
            "price":      str(g("price_fmt")),
            "mcap":       str(g("mcap_fmt")),
            "vol":        str(g("vol_fmt")),
            "score":      str(g("score_fmt")),
            "pct":        str(g("pct_fmt")),
            "ev_score":   round(float(g("ev_score", 0) or 0), 4),
            "avg_win":    round(float(g("avg_win_magnitude", 0) or 0), 2),
            "avg_loss":   round(float(g("avg_loss_magnitude", 0) or 0), 2),
            "weekly_vol": round(float(g("weekly_vol", 0) or 0), 2),
            "ev_conviction": str(g("ev_conviction", "")),
            "conviction": str(g("conviction", "low")),
            "thesis":     str(g("thesis")),
            "risk_flag":  str(g("risk_flag")),
            "thesis_src": str(g("thesis_source", "rule_based")),
            "confidence": str(g("confidence", "")),
            # Suggested stop parameters — computed from avg_win and weekly_vol
            **dict(zip(
                ["suggested_hard_stop_pct", "suggested_activation_pct", "suggested_trail_pct"],
                suggested_stops(
                    round(float(g("avg_win_magnitude", 0) or 0), 2),
                    round(float(g("weekly_vol", 0) or 0), 2)
                )
            )),
            # Signal composites
            "s_mom":  gf("sig_momentum"),
            "s_cat":  gf("sig_catalyst"),
            "s_fund": gf("sig_fundamentals"),
            "s_sent": gf("sig_sentiment"),
            # Sub-signals
            "s_rs":   gf("sig_momentum_rs"),
            "s_trd":  gf("sig_momentum_trend"),
            "s_vol":  gf("sig_momentum_vol_surge"),
            "s_brk":  gf("sig_momentum_breakout"),
            "s_earn": gf("sig_catalyst_earnings"),
            "s_ins":  gf("sig_catalyst_insider"),
            "s_anal": gf("sig_catalyst_analyst"),
            "s_grow": gf("sig_fund_growth"),
            "s_qual": gf("sig_fund_quality"),
            "s_prof": gf("sig_fund_profitability"),
            "s_news": gf("sig_sentiment_news"),
            "s_snt_anal": gf("sig_sentiment_analyst"),
            "s_short": gf("sig_sentiment_short"),
            "articles": int(g("sig_sentiment_articles", 0) or 0),
        })
    return json.dumps(rows)


def compute_kpis(df: pd.DataFrame, regime: dict) -> dict:
    conviction_counts = df["conviction"].value_counts()
    return {
        "total":       len(df),
        "very_high":   int(conviction_counts.get("very_high", 0)),
        "high":        int(conviction_counts.get("high", 0)),
        "top_sector":  df[df["conviction"].isin(["very_high","high"])]["sector"].mode().iloc[0]
                       if len(df) > 0 else "—",
        "avg_score":   round(float(df["alpha_score"].mean()), 4),
        "regime":      regime["regime"],
        "composite":   regime["composite"],
        "vix":         regime.get("context", {}).get("vix") or "—",
        "description": regime.get("description", ""),
    }


# ── HTML TEMPLATE ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MOMENTUM // {{ date }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #080c0f;
  --bg2:      #0d1318;
  --bg3:      #111820;
  --border:   #1e2d38;
  --border2:  #243545;
  --text:     #c8d8e4;
  --text2:    #6a8a9f;
  --text3:    #3d5a6b;
  --accent:   #00c8ff;
  --green:    #00e676;
  --amber:    #ffab00;
  --red:      #ff4444;
  --purple:   #b388ff;
  --mono:     'IBM Plex Mono', monospace;
  --sans:     'IBM Plex Sans', sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 13px; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
  min-height: 100vh;
  line-height: 1.5;
}

/* ── REGIME BANNER ── */
.regime-banner {
  padding: 10px 24px;
  display: flex;
  align-items: center;
  gap: 24px;
  border-bottom: 1px solid var(--border);
  font-size: 11px;
  letter-spacing: 0.08em;
}
.regime-banner.risk_on        { background: rgba(0,230,118,0.07); }
.regime-banner.trending_mixed { background: rgba(0,200,255,0.06); }
.regime-banner.choppy_neutral { background: rgba(255,171,0,0.06); }
.regime-banner.risk_off_mild  { background: rgba(255,100,0,0.07); }
.regime-banner.risk_off_severe{ background: rgba(255,68,68,0.09); }

.regime-pill {
  padding: 3px 10px;
  border-radius: 2px;
  font-weight: 600;
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.risk_on         .regime-pill { background: rgba(0,230,118,0.15); color: var(--green); border: 1px solid rgba(0,230,118,0.3); }
.trending_mixed  .regime-pill { background: rgba(0,200,255,0.12); color: var(--accent); border: 1px solid rgba(0,200,255,0.3); }
.choppy_neutral  .regime-pill { background: rgba(255,171,0,0.12);  color: var(--amber); border: 1px solid rgba(255,171,0,0.3); }
.risk_off_mild   .regime-pill { background: rgba(255,100,0,0.12);  color: #ff8c42;      border: 1px solid rgba(255,100,0,0.3); }
.risk_off_severe .regime-pill { background: rgba(255,68,68,0.12);  color: var(--red);   border: 1px solid rgba(255,68,68,0.3); }

.regime-desc { color: var(--text2); }
.regime-composite { margin-left: auto; color: var(--text2); }
.regime-composite span { color: var(--text); font-weight: 500; }

/* ── HEADER ── */
.header {
  padding: 20px 24px 0;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  border-bottom: 1px solid var(--border);
  padding-bottom: 16px;
}
.logo {
  font-size: 22px;
  font-weight: 600;
  letter-spacing: 0.15em;
  color: var(--accent);
}
.logo span { color: var(--text3); font-weight: 300; }
.datestamp { color: var(--text2); font-size: 11px; letter-spacing: 0.06em; }

/* ── KPI STRIP ── */
.kpi-strip {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  border-bottom: 1px solid var(--border);
}
.kpi {
  padding: 14px 20px;
  border-right: 1px solid var(--border);
}
.kpi:last-child { border-right: none; }
.kpi-label { font-size: 10px; color: var(--text3); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 4px; }
.kpi-value { font-size: 20px; font-weight: 500; color: var(--text); letter-spacing: -0.02em; }
.kpi-value.green  { color: var(--green); }
.kpi-value.amber  { color: var(--amber); }
.kpi-value.accent { color: var(--accent); }
.kpi-sub { font-size: 10px; color: var(--text2); margin-top: 2px; }

/* ── CONTROLS ── */
.controls {
  padding: 12px 24px;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  border-bottom: 1px solid var(--border);
  background: var(--bg2);
}
.search-wrap { position: relative; }
.search-wrap input {
  background: var(--bg3);
  border: 1px solid var(--border2);
  color: var(--text);
  font-family: var(--mono);
  font-size: 12px;
  padding: 6px 12px 6px 28px;
  width: 220px;
  outline: none;
  border-radius: 2px;
}
.search-wrap input:focus { border-color: var(--accent); }
.search-icon {
  position: absolute; left: 8px; top: 50%;
  transform: translateY(-50%);
  color: var(--text3); font-size: 12px;
}
.filter-chips {
  display: flex; gap: 6px; flex-wrap: wrap;
}
.chip {
  padding: 4px 10px;
  border: 1px solid var(--border2);
  background: transparent;
  color: var(--text2);
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.06em;
  cursor: pointer;
  border-radius: 2px;
  transition: all 0.15s;
  text-transform: uppercase;
}
.chip:hover  { border-color: var(--accent); color: var(--accent); }
.chip.active { border-color: var(--accent); background: rgba(0,200,255,0.1); color: var(--accent); }
.chip.conv-very_high.active  { border-color: var(--green);  background: rgba(0,230,118,0.1);  color: var(--green); }
.chip.conv-high.active       { border-color: #7fff7f;        background: rgba(127,255,127,0.08); color: #7fff7f; }
.chip.conv-elevated.active   { border-color: var(--amber);  background: rgba(255,171,0,0.1);  color: var(--amber); }

.controls-right { margin-left: auto; display: flex; align-items: center; gap: 8px; }
.count-badge {
  font-size: 11px; color: var(--text2);
}
.count-badge span { color: var(--accent); font-weight: 500; }

/* ── TABLE ── */
.table-wrap {
  overflow-x: auto;
  max-height: calc(100vh - 280px);
  overflow-y: auto;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
thead th {
  background: var(--bg2);
  padding: 8px 12px;
  text-align: left;
  font-weight: 500;
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text2);
  border-bottom: 1px solid var(--border2);
  cursor: pointer;
  white-space: nowrap;
  position: sticky; top: 0; z-index: 10;
  user-select: none;
}
thead th:hover { color: var(--accent); }
thead th.sorted-asc::after  { content: " ↑"; color: var(--accent); }
thead th.sorted-desc::after { content: " ↓"; color: var(--accent); }

tbody tr {
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  transition: background 0.1s;
}
tbody tr:hover    { background: var(--bg2); }
tbody tr.expanded { background: var(--bg3); }
tbody tr.detail-row td { padding: 0; }

td { padding: 7px 12px; white-space: nowrap; vertical-align: middle; }

.rank-cell   { color: var(--text3); font-size: 11px; width: 52px; }
.symbol-cell { font-weight: 600; font-size: 13px; color: var(--text); letter-spacing: 0.04em; }
.name-cell   { color: var(--text2); font-size: 11px; max-width: 160px; overflow: hidden; text-overflow: ellipsis; }
.sector-cell { color: var(--text3); font-size: 10px; letter-spacing: 0.04em; }
.price-cell  { color: var(--text); text-align: right; }
.score-cell  { color: var(--accent); font-weight: 500; text-align: right; font-size: 12px; }
.pct-cell    { color: var(--text2); font-size: 11px; text-align: right; }

/* Signal bars */
.sig-bars { display: flex; flex-direction: column; gap: 2px; min-width: 120px; }
.sig-bar-row { display: flex; align-items: center; gap: 5px; }
.sig-bar-label { font-size: 9px; color: var(--text3); width: 14px; text-align: right; letter-spacing: 0.04em; }
.sig-bar-track { flex: 1; height: 4px; background: var(--bg3); border-radius: 1px; overflow: hidden; }
.sig-bar-fill  { height: 100%; border-radius: 1px; transition: width 0.3s; }
.sig-bar-fill.mom  { background: var(--accent); }
.sig-bar-fill.cat  { background: var(--green); }
.sig-bar-fill.fund { background: var(--purple); }
.sig-bar-fill.sent { background: var(--amber); }

/* Conviction badges */
.badge {
  display: inline-block; padding: 2px 7px;
  font-size: 9px; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase;
  border-radius: 2px;
}
.badge-very_high { background: rgba(0,230,118,0.12);  color: var(--green);  border: 1px solid rgba(0,230,118,0.25); }
.badge-high      { background: rgba(127,255,127,0.08); color: #7fff7f;       border: 1px solid rgba(127,255,127,0.2); }
.badge-elevated  { background: rgba(255,171,0,0.10);   color: var(--amber);  border: 1px solid rgba(255,171,0,0.25); }
.badge-moderate  { background: rgba(100,150,180,0.08); color: var(--text2);  border: 1px solid var(--border2); }
.badge-low       { background: transparent;             color: var(--text3);  border: 1px solid var(--border); }

.llm-dot {
  display: inline-block; width: 5px; height: 5px;
  border-radius: 50%; background: var(--accent);
  margin-right: 4px; vertical-align: middle;
  box-shadow: 0 0 4px var(--accent);
}

/* ── EXPANDED DETAIL ── */
.detail-inner {
  padding: 16px 24px 20px;
  background: var(--bg3);
  border-top: 1px solid var(--border2);
  border-bottom: 2px solid var(--accent);
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr;
  gap: 20px;
}
.detail-thesis {
  grid-column: 1 / -1;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
  color: var(--text);
  line-height: 1.6;
}
.detail-thesis .label { font-size: 10px; color: var(--text3); letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px; }
.risk-flag {
  display: inline-block; margin-left: 12px;
  padding: 2px 8px; font-size: 10px;
  background: rgba(255,68,68,0.1);
  color: var(--red); border: 1px solid rgba(255,68,68,0.25);
  border-radius: 2px;
}
.detail-section .section-title {
  font-size: 10px; color: var(--text3);
  letter-spacing: 0.1em; text-transform: uppercase;
  margin-bottom: 8px; padding-bottom: 4px;
  border-bottom: 1px solid var(--border);
}
.sub-signal-grid { display: flex; flex-direction: column; gap: 5px; }
.sub-signal-row { display: flex; align-items: center; gap: 8px; }
.sub-label { font-size: 10px; color: var(--text2); width: 120px; }
.sub-bar-track { flex: 1; height: 5px; background: var(--bg); border-radius: 1px; overflow: hidden; }
.sub-bar-fill { height: 100%; border-radius: 1px; }
.sub-val { font-size: 10px; color: var(--text3); width: 36px; text-align: right; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text3); }

/* ── ANIMATIONS ── */
@keyframes fadeIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }
.detail-inner { animation: fadeIn 0.15s ease; }

/* ── FOOTER ── */
.footer {
  padding: 12px 24px;
  border-top: 1px solid var(--border);
  font-size: 10px;
  color: var(--text3);
  display: flex;
  justify-content: space-between;
  letter-spacing: 0.05em;
}
</style>
</head>
<body>

<!-- REGIME BANNER -->
<div class="regime-banner {{ regime_class }}">
  <div class="regime-pill">{{ regime_label }}</div>
  <div class="regime-desc">{{ regime_desc }}</div>
  <div class="regime-composite">
    composite <span>{{ regime_composite }}</span> &nbsp;|&nbsp;
    VIX <span>{{ vix }}</span>
  </div>
</div>

<!-- HEADER -->
<div class="header">
  <div style="display:flex;align-items:center;gap:16px">
    <a href="../index.html" style="font-size:10px;color:var(--text3);text-decoration:none;letter-spacing:.08em;border:1px solid var(--border);padding:4px 10px;border-radius:2px;white-space:nowrap">← HUB</a>
    <div>
      <div class="logo">MOMENTUM<span>//</span>ALPHA</div>
      <div class="datestamp">WEEK OF {{ date }} &nbsp;·&nbsp; {{ total }} TICKERS RANKED</div>
    </div>
  </div>
</div>

<!-- KPI STRIP -->
<div class="kpi-strip">
  <div class="kpi">
    <div class="kpi-label">Universe</div>
    <div class="kpi-value accent">{{ total }}</div>
    <div class="kpi-sub">liquid tickers</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Very High Conv.</div>
    <div class="kpi-value green">{{ very_high }}</div>
    <div class="kpi-sub">top 7% of universe</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">High Conv.</div>
    <div class="kpi-value" style="color:#7fff7f">{{ high }}</div>
    <div class="kpi-sub">top 15% of universe</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Top Sector</div>
    <div class="kpi-value" style="font-size:14px; color:var(--purple)">{{ top_sector }}</div>
    <div class="kpi-sub">by high conviction count</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Avg Alpha Score</div>
    <div class="kpi-value amber">{{ avg_score }}</div>
    <div class="kpi-sub">universe mean</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Regime Score</div>
    <div class="kpi-value">{{ regime_composite }}</div>
    <div class="kpi-sub">-1.0 bear · +1.0 bull</div>
  </div>
</div>

<!-- CONTROLS -->
<div class="controls">
  <div class="search-wrap">
    <span class="search-icon">⌕</span>
    <input type="text" id="search" placeholder="search symbol or name..." oninput="filterTable()">
  </div>

  <div class="filter-chips" id="conviction-chips">
    <button class="chip" data-filter="conviction" data-val="" onclick="setFilter('conviction','',this)">All</button>
    <button class="chip conv-very_high" data-filter="conviction" data-val="very_high" onclick="setFilter('conviction','very_high',this)">Very High</button>
    <button class="chip conv-high"      data-filter="conviction" data-val="high"      onclick="setFilter('conviction','high',this)">High</button>
    <button class="chip conv-elevated"  data-filter="conviction" data-val="elevated"  onclick="setFilter('conviction','elevated',this)">Elevated</button>
    <button class="chip conv-moderate"  data-filter="conviction" data-val="moderate"  onclick="setFilter('conviction','moderate',this)">Moderate</button>
  </div>

  <div class="filter-chips" id="sector-chips"></div>

  <div class="controls-right">
    <div class="count-badge">showing <span id="showing-count">—</span> tickers</div>
    <button onclick="exportCSV()" style="padding:4px 12px;border:1px solid var(--border2);background:transparent;color:var(--text2);font-family:var(--mono);font-size:10px;letter-spacing:.06em;cursor:pointer;border-radius:2px;transition:all .15s" onmouseover="this.style.color='var(--accent)';this.style.borderColor='rgba(0,200,255,.3)'" onmouseout="this.style.color='var(--text2)';this.style.borderColor='var(--border2)'">Export CSV</button>
  </div>
</div>

<!-- TABLE -->
<div class="table-wrap">
<table id="main-table">
<thead>
  <tr>
    <th onclick="sortTable('rank')">Rank</th>
    <th onclick="sortTable('symbol')">Symbol</th>
    <th onclick="sortTable('name')">Name</th>
    <th onclick="sortTable('sector')">Sector</th>
    <th onclick="sortTable('price')">Price</th>
    <th onclick="sortTable('mcap')">Mkt Cap</th>
    <th onclick="sortTable('score')">α Score</th>
    <th onclick="sortTable('ev_score')">EV Score</th>
    <th onclick="sortTable('avg_win')">Avg Win</th>
    <th onclick="sortTable('weekly_vol')">Wk Vol</th>
    <th onclick="sortTable('suggested_hard_stop_pct')" title="Suggested hard stop % based on weekly volatility">Hard Stop</th>
    <th onclick="sortTable('suggested_activation_pct')" title="Suggested Phase 2 activation % = 0.75 × avg win">P2 Activate</th>
    <th onclick="sortTable('suggested_trail_pct')" title="Suggested trailing stop % = 0.5 × weekly vol, max 15%">Trail %</th>
    <th onclick="sortTable('pct')">Pct Rank</th>
    <th>Signals</th>
    <th onclick="sortTable('conviction')">Conviction</th>
  </tr>
</thead>
<tbody id="table-body"></tbody>
</table>
</div>

<div class="footer">
  <span>MOMENTUM//ALPHA · Generated {{ date }} · All signals sourced from free public data</span>
  <span>· Click any row to expand signal detail · <span style="color:var(--accent)">●</span> = LLM thesis</span>
</div>

<script>
const ALL_ROWS = {{ rows_json }};
let filtered   = [...ALL_ROWS];
let sortCol    = 'rank';
let sortDir    = 'asc';
let filters    = { conviction: '', sector: '', search: '' };
let expanded   = null;

// ── CONVICTION ORDER ──────────────────────────────────────────────────────────
const CONV_ORDER = { very_high:5, high:4, elevated:3, moderate:2, low:1 };

// ── SIGNAL COLOR ─────────────────────────────────────────────────────────────
function sigColor(val) {
  if (val === null) return '#3d5a6b';
  if (val >= 0.75) return '#00e676';
  if (val >= 0.55) return '#7fff7f';
  if (val >= 0.40) return '#ffab00';
  return '#ff6b6b';
}

function fmtSig(val) {
  if (val === null || val === undefined) return '—';
  return val.toFixed(3);
}

// ── SPARKBAR HTML ─────────────────────────────────────────────────────────────
function sparkBars(row) {
  const bars = [
    { key: 's_mom',  cls: 'mom',  label: 'M' },
    { key: 's_cat',  cls: 'cat',  label: 'C' },
    { key: 's_fund', cls: 'fund', label: 'F' },
    { key: 's_sent', cls: 'sent', label: 'S' },
  ];
  return `<div class="sig-bars">${bars.map(b => {
    const v = row[b.key];
    const w = v !== null ? Math.round(v * 100) : 0;
    return `<div class="sig-bar-row">
      <span class="sig-bar-label">${b.label}</span>
      <div class="sig-bar-track">
        <div class="sig-bar-fill ${b.cls}" style="width:${w}%;background:${sigColor(v)}"></div>
      </div>
    </div>`;
  }).join('')}</div>`;
}

// ── BADGE ─────────────────────────────────────────────────────────────────────
function badge(conv) {
  return `<span class="badge badge-${conv}">${conv.replace('_',' ')}</span>`;
}

// ── SUB-SIGNAL DETAIL ─────────────────────────────────────────────────────────
function detailSection(title, subs) {
  const rows = subs.map(([label, val]) => {
    const w   = val !== null ? Math.round(val * 100) : 0;
    const col = sigColor(val);
    return `<div class="sub-signal-row">
      <span class="sub-label">${label}</span>
      <div class="sub-bar-track">
        <div class="sub-bar-fill" style="width:${w}%; background:${col}"></div>
      </div>
      <span class="sub-val" style="color:${col}">${fmtSig(val)}</span>
    </div>`;
  }).join('');
  return `<div class="detail-section">
    <div class="section-title">${title}</div>
    <div class="sub-signal-grid">${rows}</div>
  </div>`;
}

// ── EXPANDED DETAIL ROW ───────────────────────────────────────────────────────
function buildDetail(row) {
  const llmDot = row.thesis_src === 'llm' ? '<span class="llm-dot"></span>' : '';
  const flag   = row.risk_flag ? `<span class="risk-flag">⚠ ${row.risk_flag}</span>` : '';
  const conf   = row.confidence ? ` <span style="color:var(--text3);font-size:10px">[${row.confidence}]</span>` : '';

  return `<div class="detail-inner">
    <div class="detail-thesis">
      <div class="label">${llmDot}Thesis${conf}${flag}</div>
      ${row.thesis || '<em style="color:var(--text3)">No thesis generated</em>'}
    </div>
    <div class="detail-section">
      <div class="section-title">Expected value breakdown</div>
      <div class="sub-signal-grid">
        <div class="sub-signal-row">
          <span class="sub-label">EV Score</span>
          <span class="sub-val" style="color:${row.ev_score>0?'var(--green)':'var(--red)'}">${row.ev_score>0?'+':''}${row.ev_score.toFixed(4)}</span>
        </div>
        <div class="sub-signal-row">
          <span class="sub-label">α Rank → EV Rank</span>
          <span class="sub-val" style="color:var(--text2)">#${row.alpha_rank} → #${row.ev_rank}</span>
        </div>
        <div class="sub-signal-row">
          <span class="sub-label">Avg weekly win</span>
          <span class="sub-val" style="color:var(--green)">${row.avg_win>0?'+'+row.avg_win.toFixed(1)+'%':'—'}</span>
        </div>
        <div class="sub-signal-row">
          <span class="sub-label">Avg weekly loss</span>
          <span class="sub-val" style="color:var(--red)">${row.avg_loss<0?row.avg_loss.toFixed(1)+'%':'—'}</span>
        </div>
        <div class="sub-signal-row">
          <span class="sub-label">Weekly volatility</span>
          <span class="sub-val" style="color:var(--text2)">${row.weekly_vol>0?'±'+row.weekly_vol.toFixed(1)+'%':'—'}</span>
        </div>
        <div class="sub-signal-row">
          <span class="sub-label">Suggested hard stop</span>
          <span class="sub-val" style="color:var(--red);font-weight:500">${row.suggested_hard_stop_pct!=null?'-'+row.suggested_hard_stop_pct.toFixed(1)+'%':'—'}</span>
        </div>
        <div class="sub-signal-row">
          <span class="sub-label">Suggested P2 activation</span>
          <span class="sub-val" style="color:var(--amber);font-weight:500">${row.suggested_activation_pct!=null?'+'+row.suggested_activation_pct.toFixed(1)+'%':'—'}</span>
        </div>
        <div class="sub-signal-row">
          <span class="sub-label">Suggested trail %</span>
          <span class="sub-val" style="color:var(--purple);font-weight:500">${row.suggested_trail_pct!=null?row.suggested_trail_pct.toFixed(1)+'%':'—'}</span>
        </div>
      </div>
    </div>
    ${detailSection('Momentum', [
      ['RS Rank',    row.s_rs],
      ['Trend',      row.s_trd],
      ['Vol Surge',  row.s_vol],
      ['Breakout',   row.s_brk],
    ])}
    ${detailSection('Catalyst', [
      ['Earnings',   row.s_earn],
      ['Insider',    row.s_ins],
      ['Analyst',    row.s_anal],
    ])}
    ${detailSection('Fund + Sentiment', [
      ['Growth',     row.s_grow],
      ['Quality',    row.s_qual],
      ['Profitab.',  row.s_prof],
      ['News Tone',  row.s_news],
      ['Analyst Tr.',row.s_snt_anal],
      ['Short Int.', row.s_short],
    ])}
  </div>`;
}

// ── RENDER TABLE ──────────────────────────────────────────────────────────────
function renderTable() {
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';
  document.getElementById('showing-count').textContent = filtered.length.toLocaleString();

  filtered.forEach((row, i) => {
    const tr = document.createElement('tr');
    tr.dataset.idx = i;
    tr.innerHTML = `
      <td class="rank-cell">${row.rank}</td>
      <td class="symbol-cell">${row.symbol}</td>
      <td class="name-cell" title="${row.name}">${row.name}</td>
      <td class="sector-cell">${row.sector}</td>
      <td class="price-cell">${row.price}</td>
      <td class="price-cell">${row.mcap}</td>
      <td class="score-cell">${row.score}</td>
      <td class="score-cell" style="color:${row.ev_score>0?'var(--green)':row.ev_score<0?'var(--red)':'var(--text2)'}">${row.ev_score>0?'+':''}${row.ev_score.toFixed(4)}</td>
      <td style="color:var(--green);font-size:11px">${row.avg_win>0?'+'+row.avg_win.toFixed(1)+'%':'—'}</td>
      <td style="color:var(--text2);font-size:11px">${row.weekly_vol>0?'±'+row.weekly_vol.toFixed(1)+'%':'—'}</td>
      <td style="color:var(--red);font-size:11px;font-weight:500" title="Suggested hard stop % based on weekly volatility">${row.suggested_hard_stop_pct!=null?'-'+row.suggested_hard_stop_pct.toFixed(1)+'%':'—'}</td>
      <td style="color:var(--amber);font-size:11px;font-weight:500" title="Suggested Phase 2 activation: cancel hard stop, set trail">${row.suggested_activation_pct!=null?'+'+row.suggested_activation_pct.toFixed(1)+'%':'—'}</td>
      <td style="color:var(--purple);font-size:11px;font-weight:500" title="Suggested trailing stop % from high water mark">${row.suggested_trail_pct!=null?row.suggested_trail_pct.toFixed(1)+'%':'—'}</td>
      <td class="pct-cell">${row.pct}%</td>
      <td>${sparkBars(row)}</td>
      <td>${badge(row.conviction)}</td>
    `;
    tr.onclick = () => toggleDetail(tr, row);
    tbody.appendChild(tr);
  });
}

// ── EXPAND / COLLAPSE ─────────────────────────────────────────────────────────
function toggleDetail(tr, row) {
  // Close existing
  if (expanded) {
    expanded.tr.classList.remove('expanded');
    if (expanded.detailTr) expanded.detailTr.remove();
    if (expanded.tr === tr) { expanded = null; return; }
  }
  tr.classList.add('expanded');
  const detailTr = document.createElement('tr');
  detailTr.classList.add('detail-row');
  const td = document.createElement('td');
  td.colSpan = 10;
  td.innerHTML = buildDetail(row);
  detailTr.appendChild(td);
  tr.after(detailTr);
  expanded = { tr, detailTr };
}

// ── SORT ──────────────────────────────────────────────────────────────────────
function sortTable(col) {
  if (sortCol === col) {
    sortDir = sortDir === 'asc' ? 'desc' : 'asc';
  } else {
    sortCol = col;
    sortDir = col === 'rank' ? 'asc' : 'desc';
  }
  document.querySelectorAll('thead th').forEach(th => {
    th.classList.remove('sorted-asc','sorted-desc');
  });
  const headers = ['rank','symbol','name','sector','price','mcap','score','pct','signals','conviction'];
  const idx = headers.indexOf(col);
  if (idx >= 0) {
    document.querySelectorAll('thead th')[idx].classList.add(`sorted-${sortDir}`);
  }
  applySort();
  renderTable();
}

function applySort() {
  const dir = sortDir === 'asc' ? 1 : -1;
  filtered.sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol];
    if (sortCol === 'conviction') { va = CONV_ORDER[va]||0; vb = CONV_ORDER[vb]||0; }
    else if (['pct','score','ev_score','avg_win','avg_loss','weekly_vol'].includes(sortCol)) {
      va = parseFloat(va)||0; vb = parseFloat(vb)||0;
    } else if (['rank','alpha_rank','ev_rank'].includes(sortCol)) {
      va = parseInt(va)||9999; vb = parseInt(vb)||9999;
    } else {
      va = String(va||'').toLowerCase(); vb = String(vb||'').toLowerCase();
      return va < vb ? -dir : va > vb ? dir : 0;
    }
    return (va - vb) * dir;
  });
}

// ── FILTER ────────────────────────────────────────────────────────────────────
function applyFilters() {
  filtered = ALL_ROWS.filter(row => {
    if (filters.conviction && row.conviction !== filters.conviction) return false;
    if (filters.sector     && row.sector     !== filters.sector)     return false;
    if (filters.search) {
      const q = filters.search.toLowerCase();
      if (!row.symbol.toLowerCase().includes(q) &&
          !row.name.toLowerCase().includes(q)) return false;
    }
    return true;
  });
  applySort();
  renderTable();
}

function setFilter(type, val, btn) {
  filters[type] = val;
  // Update chips
  document.querySelectorAll(`[data-filter="${type}"]`).forEach(c => c.classList.remove('active'));
  btn.classList.add('active');
  applyFilters();
}

function filterTable() {
  filters.search = document.getElementById('search').value;
  applyFilters();
}

// ── EXPORT CSV ────────────────────────────────────────────────────────────────
function exportCSV() {
  const cols = [
    ['rank',                    'Composite Rank'],
    ['alpha_rank',              'Alpha Rank'],
    ['ev_rank',                 'EV Rank'],
    ['symbol',                  'Symbol'],
    ['name',                    'Name'],
    ['sector',                  'Sector'],
    ['price',                   'Price'],
    ['mcap',                    'Market Cap'],
    ['vol',                     'Avg Volume'],
    ['conviction',              'Conviction'],
    ['score',                   'Alpha Score'],
    ['ev_score',                'EV Score'],
    ['avg_win',                 'Avg Win %'],
    ['avg_loss',                'Avg Loss %'],
    ['weekly_vol',              'Weekly Vol %'],
    ['suggested_hard_stop_pct', 'Hard Stop %'],
    ['suggested_activation_pct','P2 Activation %'],
    ['suggested_trail_pct',     'Trail %'],
    ['s_mom',                   'Momentum'],
    ['s_cat',                   'Catalyst'],
    ['s_fund',                  'Fundamentals'],
    ['s_sent',                  'Sentiment'],
    ['s_rs',                    'RS Rank'],
    ['s_trd',                   'Trend'],
    ['s_vol',                   'Vol Surge'],
    ['s_brk',                   'Breakout'],
    ['s_earn',                  'Earnings'],
    ['s_ins',                   'Insider'],
    ['s_anal',                  'Analyst (Catalyst)'],
    ['s_grow',                  'Growth'],
    ['s_qual',                  'Quality'],
    ['s_prof',                  'Profitability'],
    ['s_news',                  'News Tone'],
    ['s_snt_anal',              'Analyst Trend'],
    ['s_short',                 'Short Interest'],
    ['thesis_src',              'Thesis Source'],
    ['thesis',                  'Thesis'],
    ['risk_flag',               'Risk Flag'],
  ];

  const header = cols.map(([, label]) => label).join(',');

  const rows = filtered.map(row =>
    cols.map(([key]) => {
      let v = row[key];
      if (v === null || v === undefined) v = '';
      v = String(v).replace(/\r?\n/g, ' ');
      if (v.includes(',') || v.includes('"') || v.includes("'")) {
        v = '"' + v.replace(/"/g, '""') + '"';
      }
      return v;
    }).join(',')
  );

  const csv  = [header, ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  const date = document.title.replace('MOMENTUM // ', '') || 'report';
  a.href = url;
  a.download = 'momentum_alpha_' + date + '.csv';
  a.click();
  URL.revokeObjectURL(url);
}

// ── SECTOR CHIPS ──────────────────────────────────────────────────────────────
function buildSectorChips() {
  const sectors = [...new Set(ALL_ROWS.map(r => r.sector))].sort();
  const wrap    = document.getElementById('sector-chips');
  const all     = document.createElement('button');
  all.className = 'chip active';
  all.textContent = 'All Sectors';
  all.onclick = () => setFilter('sector','',all);
  all.dataset.filter = 'sector'; all.dataset.val = '';
  wrap.appendChild(all);
  sectors.forEach(s => {
    const btn = document.createElement('button');
    btn.className = 'chip';
    btn.textContent = s.length > 14 ? s.substring(0,13)+'…' : s;
    btn.title = s;
    btn.dataset.filter = 'sector'; btn.dataset.val = s;
    btn.onclick = () => setFilter('sector', s, btn);
    wrap.appendChild(btn);
  });
}

// ── INIT ──────────────────────────────────────────────────────────────────────
buildSectorChips();
document.querySelector('[data-filter="conviction"][data-val=""]').classList.add('active');
applySort();
renderTable();
</script>
</body>
</html>"""


# ── RENDER ────────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame, regime: dict, date_str: str) -> str:
    kpis    = compute_kpis(df, regime)
    rows_js = build_rows_json(df)

    regime_name = regime["regime"]
    regime_labels = {
        "risk_on":         "RISK ON",
        "trending_mixed":  "TRENDING / MIXED",
        "choppy_neutral":  "CHOPPY / NEUTRAL",
        "risk_off_mild":   "RISK OFF — MILD",
        "risk_off_severe": "RISK OFF — SEVERE",
    }

    tmpl = Template(HTML_TEMPLATE)
    return tmpl.render(
        date=date_str,
        regime_class=regime_name,
        regime_label=regime_labels.get(regime_name, regime_name.upper()),
        regime_desc=regime.get("description", ""),
        regime_composite=f"{regime['composite']:+.3f}",
        vix=kpis["vix"],
        total=f"{kpis['total']:,}",
        very_high=kpis["very_high"],
        high=kpis["high"],
        top_sector=kpis["top_sector"],
        avg_score=f"{kpis['avg_score']:.4f}",
        rows_json=rows_js,
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(date_override: str = None):
    REPORTS.mkdir(parents=True, exist_ok=True)

    log.info("Loading final scores...")
    df = pd.read_csv(FINAL_CSV)
    log.info(f"  {len(df):,} tickers")

    log.info("Loading regime...")
    with open(REGIME_JSON) as f:
        regime = json.load(f)

    df, regime = load_and_prep()
    # date_override allows weekend refresh to write back to the Friday report
    # rather than creating a new file dated today
    date_str   = date_override if date_override else datetime.today().strftime("%Y-%m-%d")
    out_path   = REPORTS / f"{date_str}.html"

    log.info("Rendering HTML report...")
    html = render(df, regime, date_str)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = out_path.stat().st_size / 1024
    log.info(f"Report -> {out_path}  ({size_kb:.0f} KB)")

    return str(out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    run()
