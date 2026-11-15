"""
automation/shadow_lazy_veto.py
==============================
Shadow #12 (no orders), approved Nov 15, 2026: shadow #10's book (FIP-steady 12-1 momentum + chart-CNN veto, same
rules, same rebalance weeks) with one more veto -- "lazy prices" (Cohen, Malloy & Nguyen 2020; research/novel_eval.py
#3, docs "Novel edges batch"):

  lazy veto  cosine similarity of each company's latest 10-K to its previous 10-K (word counts, whole document, as
             research/fetch_10k_sim.py), 10-K accepted within the last 365 days and before the formation Friday;
             no stock in the bottom 20% of the universe's similarities ("changers"). Stocks without a value are kept.
  research   Q5-Q1 spread 95 bp/month (t 3.6) 2018-23, 129 bp (t 2.8) 2024-26; vetoing Q1 lifted the 12-1 top-decile
             book's Sharpe 0.35 -> 0.37 and 0.31 -> 0.36; corr with size 0.02, with the FIP+veto book -0.09.
  compare    against shadow #10 (data/shadow_fipveto_ledger.jsonl): the gap between the two is the veto's effect.

Similarity cache: data/lazy_prices_sims.csv (latest 10-K per company; seeded from research-data sec/tenk_sim_*,
updated every run from EDGAR's daily form index -- public SEC data, numbers only, no text stored).
State: data/shadow_lazy_state.json   Ledger: data/shadow_lazy_ledger.jsonl
"""
import datetime as dt
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import backtest as B  # noqa: E402
import fetch_10k_sim as T  # noqa: E402
import automation.shadow_fip_veto as F  # noqa: E402
from automation.shadow_lc_momentum import entry_date_this_week  # noqa: E402
from automation.shadow_regime_switch import fetch_daily, build_panel, HISTORY_DAYS  # noqa: E402
from automation.system_logger import log_event, LogStatus  # noqa: E402
from automation.notifier import notify_alert  # noqa: E402

log = logging.getLogger("shadow_lazy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
DATA = ROOT / "data"
STATE, LEDGER = DATA / "shadow_lazy_state.json", DATA / "shadow_lazy_ledger.jsonl"
CACHE, CACHE_META = DATA / "lazy_prices_sims.csv", DATA / "lazy_prices_meta.json"
LAZY_Q, MAX_AGE_DAYS = 0.20, 365


# ------------------------------------------------------------------ similarity cache (EDGAR)
def daily_10ks(day):
    """[(cik, accession)] of form '10-K' filings in EDGAR's daily form index for `day`; None if there is no index
    (a holiday, not yet published, or EDGAR unreachable)."""
    q = (day.month - 1) // 3 + 1
    raw = T.get(f"https://www.sec.gov/Archives/edgar/daily-index/{day.year}/QTR{q}/form.{day:%Y%m%d}.idx")
    if raw is None:
        return None
    out = []
    for line in raw.splitlines():
        if not line.startswith("10-K "):
            continue
        m = re.search(r"edgar/data/(\d+)/(\d{10}-\d{2}-\d{6})\.txt", line)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


def sim_for(cik, accession):
    """(tickers, accepted, sim_cos) for one 10-K vs the company's previous 10-K, or None."""
    raw = T.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if not raw:
        return None
    j = json.loads(raw)
    r = j.get("filings", {}).get("recent", {})
    rows = [dict(form=f, acc=a, doc=d, accepted=t) for f, a, d, t in
            zip(r.get("form", []), r.get("accessionNumber", []), r.get("primaryDocument", []), r.get("acceptanceDateTime", []))]
    tenks = sorted((x for x in rows if x["form"] == "10-K" and re.search(r"\.(htm|txt)", x["doc"].lower())),
                   key=lambda x: x["accepted"])
    idx = next((i for i, x in enumerate(tenks) if x["acc"] == accession), None)
    if idx is None or idx == 0:
        return None
    vecs = []
    for x in (tenks[idx - 1], tenks[idx]):
        doc = T.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{x['acc'].replace('-', '')}/{x['doc']}")
        if doc is None:
            return None
        vecs.append(T.vec(T.text_of(doc)))
    s = T.cos(vecs[1], vecs[0])
    if s is None:
        return None
    acc_et = pd.Timestamp(tenks[idx]["accepted"]).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M:%S")
    return ",".join(j.get("tickers") or []), acc_et, round(float(s), 6)


def update_cache(today):
    C = pd.read_csv(CACHE, dtype={"accession": str, "tickers": str})
    meta = json.loads(CACHE_META.read_text()) if CACHE_META.exists() else {"scanned_through": "2026-10-08"}
    day = dt.date.fromisoformat(meta["scanned_through"]) + dt.timedelta(days=1)
    known, added, failed = set(C.accession), [], 0
    while day < today:                                        # today's index may still be incomplete
        if day.weekday() < 5:
            filings = daily_10ks(day)
            if filings is None and (today - day).days <= 7:
                break                                         # not published / unreachable: retry next run
            for cik, acc in filings or []:
                if acc in known:
                    continue
                try:
                    r = sim_for(cik, acc)
                except Exception as ex:
                    r = None
                    log.warning(f"10-K {acc}: {ex}")
                if r is None:
                    failed += 1
                    continue
                added.append(dict(cik=cik, tickers=r[0], accession=acc, accepted=r[1], sim_cos=r[2]))
                known.add(acc)
        meta["scanned_through"] = day.isoformat()
        day += dt.timedelta(days=1)
    if added:
        C = pd.concat([C, pd.DataFrame(added)], ignore_index=True)
    C = C.sort_values("accepted").drop_duplicates("cik", keep="last")
    C.to_csv(CACHE, index=False)
    CACHE_META.write_text(json.dumps(meta, indent=1))
    log.info(f"10-K cache: +{len(added)} new ({failed} without a prior 10-K / unreadable); scanned through {meta['scanned_through']}")
    return C


def lazy_vetoes(C, universe, form):
    """Universe symbols in the bottom LAZY_Q of 10-K similarity at the formation Friday."""
    C = C.assign(accepted=pd.to_datetime(C.accepted))
    C = C[(C.accepted <= form) & (C.accepted > form - pd.Timedelta(days=MAX_AGE_DAYS))]
    sim = {}
    for t, s, a in zip(C.tickers.fillna(""), C.sim_cos, C.accepted):
        for x in str(t).split(","):
            x = x.strip().upper().replace("-", ".")
            if x in universe:
                sim[x] = s
                break
    s = pd.Series(sim, dtype=float)
    rk = s.rank(pct=True)
    return set(rk[rk <= LAZY_Q].index), s


# ------------------------------------------------------------------ book (shadow #10's rules + the lazy veto)
def main():
    from zoneinfo import ZoneInfo
    today = dt.datetime.now(ZoneInfo("America/Chicago")).date()
    C = update_cache(today)
    entry = entry_date_this_week(today)
    if entry is None or today < entry:
        log.info("No entry session yet this week"); return
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "holdings": {}, "entered": {}, "cash": 100.0, "last_date": None, "weeks_since_rebal": None, "nav0_spy": None}
    if state["last_date"] == entry.isoformat():
        log.info(f"Already recorded {entry} -- idempotent skip"); return
    sc = pd.read_csv(F.SCORES, low_memory=False)
    names = dict(zip(sc.symbol.astype(str), sc.get("name", sc.symbol).astype(str)))
    names.setdefault("SPY", "SPDR S&P 500 ETF Trust")
    syms = list(names) + list(state["holdings"])
    start = entry - dt.timedelta(days=max(HISTORY_DAYS, 420))
    adj = fetch_daily(syms, start, entry, "all")
    raw = fetch_daily(syms, start, entry, "raw")
    if adj.empty:
        log.warning("No bars fetched -- will retry next run"); return
    for d in (adj, raw):
        d["date"] = pd.to_datetime(d.timestamp).dt.tz_convert("America/New_York").dt.date
    if entry not in set(adj.date):
        log.warning(f"No closes for {entry} yet -- will retry next run"); return
    p = build_panel(adj, raw, {k: names.get(k, k) for k in set(syms)}, entry)
    entry_fri = pd.Timestamp(entry) + pd.Timedelta(days=4 - entry.weekday())
    form = p.weeks[p.weeks.get_loc(entry_fri) - 1]

    closes = adj.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    last = dt.date.fromisoformat(state["last_date"]) if state["last_date"] else entry
    held, exited = state["holdings"], []
    for sym in list(held):
        p0 = closes.at[last, sym] if (last in closes.index and sym in closes.columns) else np.nan
        p1 = closes.at[entry, sym] if sym in closes.columns else np.nan
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
            held[sym] *= p1 / p0
        elif not np.isfinite(p1):
            state["cash"] += held.pop(sym); exited.append(sym)
    nav = state["cash"] + sum(held.values())

    wsr = state.get("weeks_since_rebal")
    first = state["last_date"] is None
    rebal = wsr is None or wsr + 1 >= F.REBAL_WEEKS or not held
    bought, sold, info = [], [], {}
    if rebal:
        ranks, m12, fip, cnn, vetoed, n_univ, n_top = F.eligible_at(p, adj, form)
        f = B.features(p)
        U = B.universe(p, f, 5.0, 2e6).loc[form]
        lazy, sim = lazy_vetoes(C, set(U[U].index), form + pd.Timedelta(hours=23, minutes=59))
        lazy_hit = sorted(s for s in ranks.index if s in lazy)
        ranks = ranks.drop(lazy_hit).rank()
        keep = [h for h in held if h in ranks.index and ranks[h] <= F.BUFFER]
        bought = [x for x in ranks.sort_values().index if x not in keep][: F.N - len(keep)]
        names_new = keep + bought
        sold = sorted(set(held) - set(names_new))
        tgt = {s: nav / len(names_new) for s in names_new} if names_new else {}
        traded = sum(abs(tgt.get(s, 0) - held.get(s, 0)) for s in set(tgt) | set(held))
        cost = traded * F.COST_BPS / 1e4
        nav -= cost
        held = {s: v * nav / sum(tgt.values()) for s, v in tgt.items()} if tgt else {}
        state["cash"] = 0.0 if held else nav
        state["weeks_since_rebal"] = 0
        if first and F.STATE.exists():                  # align the rebalance weeks with shadow #10 for a clean pair
            s10 = json.loads(F.STATE.read_text())
            if s10.get("last_date") == entry.isoformat() and s10.get("weeks_since_rebal") is not None:
                state["weeks_since_rebal"] = s10["weeks_since_rebal"]
        info = dict(universe=n_univ, top20=n_top, eligible=int(len(ranks)), cnn_vetoed=vetoed, lazy_vetoed=lazy_hit,
                    universe_with_10k=int(len(sim)), lazy_cut=round(float(sim.quantile(LAZY_Q)), 6) if len(sim) else None,
                    turnover_pct=round(100 * traded / (nav + cost), 1),
                    holdings={s: {"mom_12_1_pct": round(float(m12[s]) * 100, 1) if s in m12.index else None,
                                  "sim_10k": float(sim[s]) if s in sim.index else None} for s in sorted(held)})
    else:
        state["cash"] = nav - sum(held.values())
        state["weeks_since_rebal"] = wsr + 1
    entered = state.get("entered", {})
    state.update(holdings=held, entered={s: entered.get(s, entry.isoformat()) for s in held}, last_date=entry.isoformat())
    spy = float(closes.at[entry, "SPY"])
    state["nav0_spy"] = state["nav0_spy"] or spy
    row = {"date": entry.isoformat(), "formation_week": str(form.date()), "rebalanced": rebal,
           "shadow_nav": round(nav, 4), "spy_index": round(100 * spy / state["nav0_spy"], 4),
           "bought": bought, "sold": sold, "exited_no_price": exited, **info}
    STATE.write_text(json.dumps(state, indent=1))
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    msg = (f"Shadow FIP+CNN+lazy {entry}: nav {row['shadow_nav']:.1f} | SPY {row['spy_index']:.1f} | "
           + (f"rebalanced: bought {len(bought)}, sold {len(sold)}, lazy-vetoed {len(info['lazy_vetoed'])}" if rebal else "held"))
    log.info(msg)
    log_event("shadow_lazy", LogStatus.SUCCESS, msg, metrics={k: row[k] for k in ("shadow_nav", "spy_index")})
    if rebal:
        notify_alert("shadow_lazy", msg)


if __name__ == "__main__":
    main()
