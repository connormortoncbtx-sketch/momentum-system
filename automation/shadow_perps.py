"""
automation/shadow_perps.py -- Shadow #perps (no orders): the BTC funding carry, marked forward from Oct 27, 2026.

The trade being watched (research/perps_funding.py, perps_edge.py): SHORT Kalshi's BTC perpetual (KXBTCPERP), hold
the same notional of spot BTC, collect funding (positive rate = longs pay shorts). Backtest: Kalshi's BTC funding
ran ~6-7 points/yr richer than Deribit/OKX; the carry was ~7-11%/yr on capital, 3-7 points over T-bills -- on a
market only a few months old. This marks it live, on paper, so January's review has an honest out-of-sample number.

Per funding settlement since START (public endpoint, no keys):
  funding_usd   NOTIONAL x rate (what the short perp collects; negative when it pays)
  basis_usd     the hedged pair's price P&L: -(perp mark change - spot change) x BTC held, spot = Alpaca BTC/USD
                hourly close at the funding time (zero if spot is missing for either end)
Capital assumed = spot notional + MARGIN of the perp notional (Kalshi's actual margin requirement to be confirmed);
benchmark = T-bills on that capital at CASH_YR. Idempotent: re-running only adds settlements not yet in the ledger.
Files: data/shadow_perps_ledger.jsonl (one row per funding), data/shadow_perps_state.json (running totals).
"""
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LEDGER, STATE = DATA / "shadow_perps_ledger.jsonl", DATA / "shadow_perps_state.json"
API = "https://external-api.kalshi.com/trade-api/v2"
TICKER = "KXBTCPERP"
START = pd.Timestamp("2026-10-27", tz="UTC")
NOTIONAL = 10_000.0
MARGIN = 0.20
CASH_YR = 0.038


def get(path, **q):
    url = API + path + ("?" + urllib.parse.urlencode(q) if q else "")
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            if i == 3:
                raise
            time.sleep(2 + 3 * i)


def funding():
    rows, cur = [], None
    for _ in range(50):
        q = {"ticker": TICKER, "limit": 1000}
        if cur:
            q["cursor"] = cur
        d = get("/margin/funding_rates/historical", **q)
        fr = d.get("funding_rates") or []
        rows += [r for r in fr if r.get("market_ticker", TICKER) == TICKER]
        cur = d.get("cursor")
        if not cur or not fr:
            break
    f = pd.DataFrame(rows)
    if f.empty:
        return f
    f["t"] = pd.to_datetime(f.funding_time, utc=True)
    f["rate"] = pd.to_numeric(f.funding_rate, errors="coerce")
    f["mark"] = pd.to_numeric(f.get("mark_price"), errors="coerce")
    return f.dropna(subset=["rate"]).drop_duplicates("t").sort_values("t")


def spot(t0):
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame
        b = CryptoHistoricalDataClient().get_crypto_bars(CryptoBarsRequest(
            symbol_or_symbols="BTC/USD", timeframe=TimeFrame.Hour, start=t0.to_pydatetime())).df.reset_index()
        b["t"] = pd.to_datetime(b.timestamp, utc=True) + pd.Timedelta("1h")
        return b.drop_duplicates("t", keep="last").set_index("t").close.sort_index()
    except Exception as e:
        print(f"::warning::BTC spot unavailable ({type(e).__name__}); basis left at 0")
        return pd.Series(dtype=float)


def at(s, t):
    if s.empty:
        return None
    s = s[s.index <= t]
    return float(s.iloc[-1]) if len(s) and (t - s.index[-1]) <= pd.Timedelta("2h") else None


def main():
    DATA.mkdir(exist_ok=True)
    done = set()
    last = None
    if LEDGER.exists():
        for line in LEDGER.open():
            r = json.loads(line)
            done.add(r["t"]); last = r
    f = funding()
    if f.empty:
        print("::warning::no funding data returned"); return
    f = f[f.t >= START]
    new = f[~f.t.map(lambda x: x.isoformat()).isin(done)]
    sp = spot(START - pd.Timedelta("3h"))
    rows = []
    prev_mark, prev_spot = (last or {}).get("mark"), (last or {}).get("spot")
    for r in new.itertuples():
        s = at(sp, r.t)
        mark = None if pd.isna(r.mark) else float(r.mark)
        if mark is not None and s and abs(mark / s - 1) > 0.05:
            mark = None                                # not quoted in $ per BTC -- don't guess its units
        ref = mark or s
        qty = NOTIONAL / ref if ref else None
        basis = 0.0
        if qty and None not in (prev_mark, prev_spot, mark, s):
            basis = -((mark - prev_mark) - (s - prev_spot)) * qty
        row = dict(t=r.t.isoformat(), rate=float(r.rate), funding_usd=NOTIONAL * float(r.rate), basis_usd=basis,
                   mark=mark, spot=s)
        rows.append(row)
        prev_mark, prev_spot = (mark, s) if None not in (mark, s) else (prev_mark, prev_spot)
    with LEDGER.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    L = pd.DataFrame([json.loads(x) for x in LEDGER.open()]) if LEDGER.exists() else pd.DataFrame()
    if L.empty:
        print("::notice::shadow perps: no settlements since start yet"); return
    L["t"] = pd.to_datetime(L.t, utc=True)
    days = max((L.t.max() - START).total_seconds() / 86400, 1 / 3)
    cap = NOTIONAL * (1 + MARGIN)
    fund, basis = L.funding_usd.sum(), L.basis_usd.sum()
    bill = cap * CASH_YR * days / 365
    st = dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), start=START.isoformat(),
              settlements=len(L), days=round(days, 2), notional=NOTIONAL, capital=cap, funding_usd=round(fund, 2),
              basis_usd=round(basis, 2), tbill_usd=round(bill, 2),
              carry_ann_pct=round(fund / cap * 365 / days * 100, 2),
              total_ann_pct=round((fund + basis) / cap * 365 / days * 100, 2),
              excess_ann_pct=round((fund + basis - bill) / cap * 365 / days * 100, 2),
              share_positive=round(float((L.rate > 0).mean()), 3), last_rate=float(L.rate.iloc[-1]))
    STATE.write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))
    print(f"::notice::shadow perps ({st['days']:.1f}d, {len(L)} fundings, +{len(rows)} new): funding ${fund:+.2f} "
          f"basis ${basis:+.2f} vs T-bills ${bill:.2f} on ${cap:,.0f} -> carry {st['carry_ann_pct']:+.1f}%/yr, "
          f"all-in {st['total_ann_pct']:+.1f}%/yr, over bills {st['excess_ann_pct']:+.1f} pts; "
          f"{st['share_positive']:.0%} of fundings paid the short")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc(); print(tb); print("::error::" + tb[-900:].replace("\n", " | ")); sys.exit(1)
