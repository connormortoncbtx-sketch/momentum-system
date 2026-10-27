"""
automation/live_stack.py
========================
REAL-MONEY momentum v2 (the shadow #7 v2 "stack") in Connor's live Alpaca account, with features that switch on as the
account grows (Connor, Oct 25, 2026). The design and weights are shadow #7 v2's (automation/shadow_stack.py, frozen on
2017-23 data); this file only turns its target into orders and scales it to the account:

  sleeves   SLOW 10.4%   shadow #4's slow 12-1 momentum holdings (data/shadow_slow_state.json), equal weight
            CONTRACT 23.8%  shadow #6's diversified contract-win book (data/shadow_filings_state.json, open
                         long_contract_div positions, weights 1/vol60 capped at 25%, held 10 sessions)
            ETFTREND 50.4%  41 ETFs, 63-day trend, 10% vol target each, month-end weights, capped at 1x
            CRYPTO 15.4%    16 coins above their 50-day average, 40% vol target each, capped at 1x -- held through
                         spot ETFs (IBIT/ETHA/BSOL/XRP) since crypto isn't available in Texas; other coins = cash
  sizing    vol target: leverage = 0.020698 / sd(last 8 weekly stack returns) from shadow #7's own history

  TIERS (decided from the live account each run; hysteresis so it doesn't flip back and forth)
  LITE   equity < $2,000 or a cash account: long-only, NO hedge, leverage capped at 1x (vol target may scale DOWN),
         fractional shares (stocks Alpaca can't fraction: whole shares if one share fits, else skipped and logged)
  FULL   margin account with equity >= $2,000 (drops back below $1,800): adds the IWM hedge (short, whole shares,
         0.798 x SLOW + 0.743 x CONTRACT exposure) and leverage up to LEV_CAP (default 2.0; the backtest allowed 3.0)
  CONDOR gated: only reported ("eligible at N contracts") until Connor approves after the January paper review
         (CONDOR_LIVE=yes); the live condor itself is not built yet.

Safety
  - LIVE_STACK_MODE (GitHub variable) must be "live" to place orders; anything else = dry run (plans and notifies only)
  - the account is assumed DEDICATED: every position in it is managed (anything not in the target is sold)
  - trades only when the gap to target is > max($1, 10% of the position) -- no churning a small account
  - total long exposure capped at 95% of Alpaca's buying power multiplier; crypto is never on margin
  - the repo is PUBLIC: dollar amounts and positions go to the phone (ntfy) only, never to the Actions log or git
Keys: ALPACA_LIVE_KEY / ALPACA_LIVE_SECRET (GitHub secrets). Runs daily ~10:15 ET (live_stack.yml).
"""
import datetime as dt
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import automation.shadow_stack as SS  # noqa: E402

log = logging.getLogger("live_stack")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
DATA = ROOT / "data"
STATE = DATA / "live_stack_state.json"            # tier only -- no dollar amounts (public repo)
SLOW_STATE, FILINGS_STATE, STACK_STATE = DATA / "shadow_slow_state.json", DATA / "shadow_filings_state.json", DATA / "shadow_stack_state.json"
W2, BETA = SS.W2, SS.BETA
FULL_ON, FULL_OFF = 2000.0, 1800.0
LEV_CAP = float(os.environ.get("LEV_CAP", "2.0"))
LITE_LEV_CAP = 1.0
MIN_TRADE, BAND = 1.0, 0.10      # Alpaca's $1 minimum; then only re-trade a gap > 10% of the position
CAP_DIV = 0.25
# Connor's live account (Texas) can't trade crypto (Alpaca offers it only in approved states; no crypto agreement
# exists on the account -- Oct 27, 2026). The crypto sleeve is held through US spot ETFs instead: same coin signals
# and weights; a coin without a tradable ETF stays in cash (Connor's choice, "faithful" -- research/crypto4.py tests
# the 4-coin renormalised alternative for January). CRYPTO_VIA=coins restores direct coins.
CRYPTO_VIA = (os.environ.get("CRYPTO_VIA") or "etf").strip().lower()
CRYPTO_ETF = {"BTC/USD": ["IBIT", "FBTC"], "ETH/USD": ["ETHA", "FETH"], "SOL/USD": ["BSOL", "FSOL", "GSOL"],
              "XRP/USD": ["XRP", "GXRP", "XRPZ"]}
CONDOR_MAXLOSS_SPY2, CONDOR_RISK = 170.0, 0.04   # SPY $2-wide condor, ~max loss per contract; risk per day at ~$4-5k


# ------------------------------------------------------------------ sleeve targets (fractions of the sleeve)
def slow_weights():
    s = json.loads(SLOW_STATE.read_text()) if SLOW_STATE.exists() else {}
    h = [k for k, v in (s.get("holdings") or {}).items() if v]
    return {k: 1.0 / len(h) for k in h} if h else {}


def contract_weights():
    s = json.loads(FILINGS_STATE.read_text()) if FILINGS_STATE.exists() else {}
    pos = [o for o in s.get("open", []) if o.get("book") == "long_contract_div" and o.get("entry_date")]
    if not pos:
        return {}
    iv = {o["symbol"]: 1.0 / max(float(o.get("vol60") or 0.8), 0.2) for o in pos}
    tot = sum(iv.values())
    return {k: min(v / tot, CAP_DIV) for k, v in iv.items()}


def etf_weights(px, today):
    """ETF trend weights set at the last month-end close before today (research/trend_test.py rules, capped at 1x)."""
    px = px[px.index < pd.Timestamp(today)]
    ret = px.pct_change(); vol = ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    me = [px.index[px.index <= m][-1] for m in px.resample("ME").last().index if (px.index <= m).any()]
    me = [d for d in me if d.month != pd.Timestamp(today).month or d.year != pd.Timestamp(today).year]
    d = me[-1]
    sig = np.sign(px.loc[d] / px.shift(63).loc[d] - 1).clip(lower=0)
    tgt = (sig * (0.10 / vol.loc[d]).clip(upper=2.0)).fillna(0)
    tgt = tgt / max((tgt != 0).sum(), 1) * 4
    if tgt.sum() > 1:
        tgt = tgt / tgt.sum()
    return {k: float(v) for k, v in tgt.items() if v > 0}


def crypto_weights(cp, today):
    """Coins above their 50-day average at the last complete daily close, 40% vol target each, capped at 1x."""
    cp = cp[cp.index < pd.Timestamp(today)]
    cr = cp.pct_change(); cv = cr.rolling(30, min_periods=20).std() * np.sqrt(365)
    last = cp.iloc[-1]; ma = cp.rolling(50, min_periods=50).mean().iloc[-1]
    w = ((last > ma).astype(float) * (0.40 / cv.iloc[-1]).clip(upper=1.5)).where(last.notna()).dropna()
    w = w / max(int(last.notna().sum()), 1)
    if w.sum() > 1:
        w = w / w.sum()
    return {k: float(v) for k, v in w.items() if v > 0}


def vol_target_leverage():
    s = json.loads(STACK_STATE.read_text()) if STACK_STATE.exists() else {}
    h = (s.get("v2") or {}).get("hist") or SS.VT_SEED
    sd = float(np.std(h[-8:], ddof=1))
    return float(np.clip(SS.VT_K / sd, SS.VT_LO, SS.VT_HI)) if sd > 0 else 1.0


# ------------------------------------------------------------------ tier + combined target
def decide_tier(equity, multiplier, prev):
    if multiplier < 2:
        return "LITE"
    if prev == "FULL":
        return "FULL" if equity >= FULL_OFF else "LITE"
    return "FULL" if equity >= FULL_ON else "LITE"


def resolve_crypto_etfs(tc):
    """{coin: ETF ticker} for the coins with a tradable ETF in this account (first candidate that is active+tradable)."""
    out = {}
    for coin, cands in CRYPTO_ETF.items():
        for t in cands:
            try:
                a = tc.get_asset(t)
                if a.tradable and str(a.status).lower().endswith("active") and "equity" in str(a.asset_class).lower():
                    out[coin] = t; break
            except Exception:
                continue
    return out


def targets(equity, tier, sleeves, lev_vt, multiplier, coin_etf=None):
    """{symbol: target $} (negative = short) and a summary. sleeves: {name: {symbol: weight within sleeve}}.
    coin_etf: hold each coin through this ETF instead (coins not in it go to cash)."""
    lev = min(lev_vt, LEV_CAP if tier == "FULL" else LITE_LEV_CAP)
    tgt = {}
    for name, sw in (("SLOW_h", sleeves["SLOW"]), ("CONTRACT_h", sleeves["CONTRACT"]), ("ETFTREND", sleeves["ETF"]),
                     ("CRYPTO", sleeves["CRYPTO"])):
        for sym, w in sw.items():
            tgt[sym] = tgt.get(sym, 0.0) + equity * lev * W2[name] * w
    hedge = 0.0
    if tier == "FULL":
        inv_slow, inv_con = sum(sleeves["SLOW"].values()), sum(sleeves["CONTRACT"].values())
        hedge = -equity * lev * (W2["SLOW_h"] * BETA["SLOW_h"] * inv_slow + W2["CONTRACT_h"] * BETA["CONTRACT_h"] * inv_con)
        tgt["IWM"] = tgt.get("IWM", 0.0) + hedge
    if coin_etf is not None:
        for k in [k for k in tgt if "/" in k]:
            v = tgt.pop(k)
            if k in coin_etf:
                tgt[coin_etf[k]] = tgt.get(coin_etf[k], 0.0) + v
    # what the account can carry: crypto is paid in cash (never on margin); stocks (long + short) up to 95% of the
    # margin multiplier x the equity left after crypto (Reg-T 50% on both sides). LITE: no margin at all.
    cry = sum(v for k, v in tgt.items() if "/" in k)
    stk = sum(abs(v) for k, v in tgt.items() if "/" not in k)
    room = 0.95 * (min(multiplier, 2.0) if tier == "FULL" else 1.0) * max(equity - cry, 0.0)
    scale = min(1.0, room / stk) if stk > 0 else 1.0
    tgt = {k: (v * scale if "/" not in k else v) for k, v in tgt.items()}
    return tgt, dict(lev=lev, lev_vt=lev_vt, stock_scale=scale, hedge=hedge * scale)


def plan_orders(tgt, cur, px, fractionable):
    """[(symbol, side, kind, amount)] kind 'notional' ($) or 'qty' (shares/coins). cur: {symbol: $ market value}.
    Stocks Alpaca can't fraction trade in whole shares (rounded to the nearest share of the TARGET)."""
    out, skipped = [], []
    for sym in sorted(set(tgt) | set(cur)):
        t, c = tgt.get(sym, 0.0), cur.get(sym, 0.0)
        p = px.get(sym)
        if t == 0 and abs(c) > 0.5:                           # not in the target any more: close it, price or not
            out.append((sym, "sell" if c > 0 else "buy", "all", 0.0)); continue
        if not p or p <= 0:
            if abs(t - c) > MIN_TRADE:
                skipped.append((sym, "no price"))
            continue
        crypto = "/" in sym
        if not crypto and (t < 0 or c < 0 or not fractionable.get(sym, False)):
            tq, cq = math.floor(abs(t) / p + 0.5), round(c / p)   # whole shares (shorts are always whole shares)
            if tq * p > 1.5 * abs(t):                          # don't overshoot a small target by more than 50%
                tq = math.floor(abs(t) / p)
            tq = tq if t >= 0 else -tq
            if t > 0 and tq == 0:
                skipped.append((sym, f"one share ${p:,.2f} > target ${t:,.2f}, not fractionable"))
            if tq != cq:
                out.append((sym, "buy" if tq > cq else "sell", "qty", float(abs(tq - cq))))
            continue
        d = t - c
        if abs(d) <= max(MIN_TRADE, BAND * max(abs(t), abs(c))) and not (t == 0 and c > 1.0):
            continue
        if d > 0:
            out.append((sym, "buy", "notional", round(d, 2)))
        else:
            q = abs(d) / p if t > 0 else None                 # full exit: sell the whole position
            out.append((sym, "sell", "qty" if q else "all", round(q, 6) if q else 0.0))
    out.sort(key=lambda o: o[1] != "sell")                      # sells first: frees cash for the buys
    return out, skipped


# ------------------------------------------------------------------ broker
def client():
    from alpaca.trading.client import TradingClient
    return TradingClient(os.environ["ALPACA_LIVE_KEY"], os.environ["ALPACA_LIVE_SECRET"], paper=False)


def prices(syms):
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest, CryptoLatestTradeRequest
    from alpaca.data.enums import DataFeed
    out = {}
    st = [s for s in syms if "/" not in s]; cr = [s for s in syms if "/" in s]
    k, sec = os.environ["ALPACA_LIVE_KEY"], os.environ["ALPACA_LIVE_SECRET"]
    if st:
        r = StockHistoricalDataClient(k, sec).get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=st, feed=DataFeed.IEX))
        out.update({s: float(v.price) for s, v in r.items()})
    if cr:
        r = CryptoHistoricalDataClient().get_crypto_latest_trade(CryptoLatestTradeRequest(symbol_or_symbols=cr))
        out.update({s: float(v.price) for s, v in r.items()})
    return out


def submit(tc, sym, side, kind, amt, market_open):
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    crypto = "/" in sym
    if not crypto and not market_open:
        return "skipped: market closed"
    if kind == "all":
        tc.close_position(sym.replace("/", "") if crypto else sym)
        return "closed"
    kw = dict(symbol=sym, side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
              time_in_force=TimeInForce.GTC if crypto else TimeInForce.DAY)
    kw["notional" if kind == "notional" else "qty"] = amt
    o = tc.submit_order(MarketOrderRequest(**kw))
    return str(o.status)


def main():
    from automation.notifier import notify_success, notify_alert
    if not os.environ.get("ALPACA_LIVE_KEY"):
        print("::notice::live stack: no ALPACA_LIVE_KEY secret -- skipped"); return
    mode = (os.environ.get("LIVE_STACK_MODE") or "dry").strip().lower()
    live = mode == "live"
    today = dt.date.today()
    tc = client()
    acct = tc.get_account()
    equity, mult = float(acct.equity), float(acct.multiplier)
    st = json.loads(STATE.read_text()) if STATE.exists() else {"tier": None}
    tier = decide_tier(equity, mult, st.get("tier"))
    start = today - dt.timedelta(days=420)
    sleeves = {"SLOW": slow_weights(), "CONTRACT": contract_weights(),
               "ETF": etf_weights(SS.etf_daily(start), today)}
    cp = SS.crypto_daily(start)
    sleeves["CRYPTO"] = crypto_weights(cp, today)
    _c = cp[cp.index < pd.Timestamp(today)]
    print(f"::notice::crypto sleeve: {len(sleeves['CRYPTO'])} of {_c.shape[1]} coins above their 50-day average "
          f"(last daily close {_c.index[-1]:%Y-%m-%d}); ETF sleeve: {len(sleeves['ETF'])} of {len(SS.ETFS)} ETFs in an uptrend")
    lev_vt = vol_target_leverage()
    market_open = bool(tc.get_clock().is_open)
    coin_etf = resolve_crypto_etfs(tc) if CRYPTO_VIA == "etf" else None
    if coin_etf is not None:
        held = [c for c in sleeves["CRYPTO"] if c in coin_etf]
        print(f"::notice::crypto via ETFs: {len(held)} of {len(sleeves['CRYPTO'])} trending coins have a tradable ETF "
              f"({', '.join(sorted(coin_etf.values()))} available); the rest of the sleeve stays in cash")
        if not market_open:
            print("::notice::market closed and no direct crypto -- nothing to trade"); return
    tgt, info = targets(equity, tier, sleeves, lev_vt, mult, coin_etf)
    pos = tc.get_all_positions()
    norm = lambda p: (p.symbol[:-3] + "/USD") if "crypto" in str(p.asset_class).lower() and "/" not in p.symbol else p.symbol
    cur = {norm(p): float(p.market_value) for p in pos}
    px = prices(sorted(set(tgt) | set(cur)))
    frac = {}
    for s in [s for s in tgt if "/" not in s]:
        try:
            frac[s] = bool(tc.get_asset(s).fractionable)
        except Exception:
            frac[s] = False
    orders, skipped = plan_orders(tgt, cur, px, frac)
    results = []
    for sym, side, kind, amt in orders:
        if not live:
            results.append((sym, side, kind, amt, "dry run")); continue
        try:
            results.append((sym, side, kind, amt, submit(tc, sym, side, kind, amt, market_open)))
        except Exception as e:
            results.append((sym, side, kind, amt, f"error: {str(e)[:120]}"))
    condor_n = math.floor(CONDOR_RISK * equity / CONDOR_MAXLOSS_SPY2)
    condor = ("gated (approve after the January review)" if os.environ.get("CONDOR_LIVE", "no") != "yes" else "approved -- not built yet") \
        + (f"; eligible for {condor_n} SPY $2-wide contract(s)/day at 4% risk" if condor_n >= 1 else "; not yet eligible (~$4,250 for 1 contract)")
    # public log: no dollars, no positions
    print(f"::notice::live stack ({mode}): tier {tier}, leverage {info['lev']:.2f}x (vol target {info['lev_vt']:.2f}), "
          f"stock scale {info['stock_scale']:.2f}, {len(orders)} orders planned, {len(skipped)} skipped, errors {sum('error' in r[4] for r in results)}; condor {condor.split(';')[0]}")
    if tier != st.get("tier") and st.get("tier") is not None:
        notify_alert("Live stack", f"TIER CHANGE {st.get('tier')} -> {tier} (equity ${equity:,.2f}, margin x{mult:g})")
    STATE.write_text(json.dumps({"tier": tier, "last_run": today.isoformat(), "mode": mode}, indent=1))
    lines = [f"{'LIVE' if live else 'DRY RUN'} {today}: equity ${equity:,.2f}, tier {tier}, leverage {info['lev']:.2f}x"
             + (f", IWM hedge ${info['hedge']:,.0f}" if tier == "FULL" else "")]
    lines += [f"{s} {sd} {('$' + format(a, ',.2f')) if k == 'notional' else (format(a, 'g') if k == 'qty' else 'ALL')} -> {r}"
              for s, sd, k, a, r in results]
    lines += [f"skipped {s}: {why}" for s, why in skipped]
    lines.append(f"condor: {condor}")
    msg = "\n".join(lines)
    (notify_alert if any("error" in r[4] for r in results) else notify_success)("Live stack", msg[:3500])


if __name__ == "__main__":
    main()
