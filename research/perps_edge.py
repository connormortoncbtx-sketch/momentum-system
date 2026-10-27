"""
research/perps_edge.py -- is there a structural inefficiency in Kalshi's perps? (Oct 27, 2026, Connor)
Follow-up to perps_funding.py (carry alone: BTC +9.4%/yr, thin). Two tests on public data:

1. FUNDING vs OFFSHORE: Kalshi's BTC funding (8-hourly) against the big offshore venues on the same days -- Deribit,
   Hyperliquid, OKX, Bybit (whichever answer from a US runner). If Kalshi runs persistently richer, US-only retail longs
   are overpaying on a fenced-off venue: a structural premium (like 0DTE), harvestable by short Kalshi perp + long spot,
   or short Kalshi / long offshore where accessible. Reported: annualised funding per venue over the common days, the
   Kalshi-minus-venue spread, its t-stat (daily), share of days Kalshi is richer.
2. BASIS / MEAN REVERSION: Kalshi BTC perp mid (hourly candles) vs BTC spot (hourly, Alpaca). Premium is measured as the
   ratio's deviation from its own rolling 7-day median (sidesteps any contract multiplier). Distribution of the
   premium, how fast it decays (AR(1) half-life), and a simple hedged reversion rule: when |premium| > k bps, take the
   other side (perp vs spot) and close when it crosses 0 -- gross bps per trade and trades/month, vs fees.
Aggregates only -> results/perps_edge.csv.
"""
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
KAL = "https://external-api.kalshi.com/trade-api/v2"
pd.set_option("display.width", 220)
UA = {"Accept": "application/json", "User-Agent": "momentum-research/1.0"}


def get(url, data=None):
    for i in range(3):
        try:
            req = urllib.request.Request(url, headers={**UA, **({"Content-Type": "application/json"} if data else {})},
                                         data=json.dumps(data).encode() if data else None)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            err = e
            time.sleep(2 + 2 * i)
    raise err


def kalshi_funding():
    d = get(f"{KAL}/margin/funding_rates/historical?" + urllib.parse.urlencode({"ticker": "KXBTCPERP"}))
    f = pd.DataFrame(d.get("funding_rates") or [])
    f["t"] = pd.to_datetime(f.funding_time, utc=True)
    f["rate"] = pd.to_numeric(f.funding_rate, errors="coerce")
    return f[["t", "rate"]].dropna().sort_values("t")


def deribit(t0):
    # interest_8h per hour bucket; daily funding = mean(interest_8h) * 3
    rows, end = [], int(time.time() * 1000)
    start = int(t0.timestamp() * 1000)
    while end > start:
        s = max(start, end - 30 * 86400 * 1000)
        d = get("https://www.deribit.com/api/v2/public/get_funding_rate_history?" + urllib.parse.urlencode(
            {"instrument_name": "BTC-PERPETUAL", "start_timestamp": s, "end_timestamp": end}))
        rows += d.get("result", [])
        end = s - 1
    x = pd.DataFrame(rows)
    x["t"] = pd.to_datetime(x.timestamp, unit="ms", utc=True)
    x["rate"] = x.interest_8h / 8                       # per hour
    return x[["t", "rate"]]


def hyperliquid(t0):
    rows, s = [], int(t0.timestamp() * 1000)
    for _ in range(60):
        d = get("https://api.hyperliquid.xyz/info", {"type": "fundingHistory", "coin": "BTC", "startTime": s})
        if not d:
            break
        rows += d
        s2 = int(d[-1]["time"]) + 1
        if s2 <= s or len(d) < 400:
            break
        s = s2
    x = pd.DataFrame(rows)
    x["t"] = pd.to_datetime(x.time, unit="ms", utc=True)
    x["rate"] = pd.to_numeric(x.fundingRate)            # hourly
    return x[["t", "rate"]]


def okx(t0):
    rows, after = [], None
    for _ in range(40):
        q = {"instId": "BTC-USDT-SWAP", "limit": 100}
        if after:
            q["after"] = after
        d = get("https://www.okx.com/api/v5/public/funding-rate-history?" + urllib.parse.urlencode(q)).get("data", [])
        if not d:
            break
        rows += d
        after = d[-1]["fundingTime"]
        if int(after) < t0.timestamp() * 1000:
            break
    x = pd.DataFrame(rows)
    x["t"] = pd.to_datetime(pd.to_numeric(x.fundingTime), unit="ms", utc=True)
    x["rate"] = pd.to_numeric(x.realizedRate if "realizedRate" in x else x.fundingRate)
    return x[["t", "rate"]]


def bybit(t0):
    rows, end = [], int(time.time() * 1000)
    for _ in range(40):
        d = get("https://api.bybit.com/v5/market/funding/history?" + urllib.parse.urlencode(
            {"category": "linear", "symbol": "BTCUSDT", "limit": 200, "endTime": end})).get("result", {}).get("list", [])
        if not d:
            break
        rows += d
        end = int(d[-1]["fundingRateTimestamp"]) - 1
        if end < t0.timestamp() * 1000:
            break
    x = pd.DataFrame(rows)
    x["t"] = pd.to_datetime(pd.to_numeric(x.fundingRateTimestamp), unit="ms", utc=True)
    x["rate"] = pd.to_numeric(x.fundingRate)
    return x[["t", "rate"]]


def daily(x):
    return x.set_index("t").rate.groupby(lambda z: z.date()).sum()


def kalshi_candles(t0):
    out, s = [], int(t0.timestamp())
    now = int(time.time())
    while s < now:
        e = min(now, s + 20 * 86400)
        d = get(f"{KAL}/margin/markets/KXBTCPERP/candlesticks?" + urllib.parse.urlencode(
            {"start_ts": s, "end_ts": e, "period_interval": 60}))
        for c in d.get("candlesticks", []):
            b = pd.to_numeric((c.get("bid") or {}).get("close"), errors="coerce")
            a = pd.to_numeric((c.get("ask") or {}).get("close"), errors="coerce")
            out.append((c.get("end_period_ts"), b, a, pd.to_numeric(c.get("volume"), errors="coerce")))
        s = e
    k = pd.DataFrame(out, columns=["ts", "bid", "ask", "vol"]).dropna(subset=["bid", "ask"])
    k = k[(k.bid > 0) & (k.ask >= k.bid)]
    k["t"] = pd.to_datetime(k.ts, unit="s", utc=True)
    k["mid"] = (k.bid + k.ask) / 2
    k["spread_bps"] = (k.ask - k.bid) / k.mid * 1e4
    k = k.drop_duplicates("ts", keep="last")
    return k.set_index("t").sort_index()


def btc_spot(t0):
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    b = CryptoHistoricalDataClient().get_crypto_bars(CryptoBarsRequest(symbol_or_symbols="BTC/USD", timeframe=TimeFrame.Hour,
                                                                      start=t0)).df.reset_index()
    b["t"] = pd.to_datetime(b.timestamp, utc=True) + pd.Timedelta("1h")        # bar end, like Kalshi's end_period_ts
    b = b.drop_duplicates("t", keep="last")
    return b.set_index("t").close.sort_index()


def main():
    res = []
    kf = kalshi_funding()
    t0 = kf.t.min()
    kd = daily(kf)
    print(f"Kalshi BTC funding: {len(kf)} settlements from {t0:%Y-%m-%d}")
    venues = {}
    for name, fn in (("deribit", deribit), ("hyperliquid", hyperliquid), ("okx", okx), ("bybit", bybit)):
        try:
            venues[name] = daily(fn(t0))
            print(f"  {name}: {len(venues[name])} days")
        except Exception as e:
            print(f"::warning::{name} funding unavailable: {str(e)[:120]}")
    for name, vd in venues.items():
        j = pd.concat([kd.rename("k"), vd.rename("v")], axis=1).dropna()
        j = j.iloc[1:-1] if len(j) > 3 else j
        if len(j) < 10:
            continue
        sp = j.k - j.v
        r = dict(test="funding_vs_offshore", venue=name, days=len(j), kalshi_ann_pct=j.k.mean() * 365 * 100,
                 venue_ann_pct=j.v.mean() * 365 * 100, spread_ann_pct=sp.mean() * 365 * 100,
                 t=sp.mean() / sp.std() * np.sqrt(len(sp)) if sp.std() > 0 else np.nan, kalshi_richer_share=(sp > 0).mean(),
                 last30_spread_ann_pct=sp.iloc[-30:].mean() * 365 * 100, corr=j.k.corr(j.v))
        res.append(r)
    # basis
    try:
        k = kalshi_candles(t0)
        sp = btc_spot(t0 - pd.Timedelta("2h"))
        j = pd.concat([k.mid, k.spread_bps, k.vol, sp.rename("spot")], axis=1, join="inner").dropna()
        ratio = j.mid / j.spot
        base = ratio.rolling(24 * 7, min_periods=24).median()
        prem = ((ratio / base - 1) * 1e4).dropna()                       # bps vs its own 7-day median
        ac = prem.autocorr(1)
        hl = -np.log(2) / np.log(ac) if 0 < ac < 1 else np.nan
        res.append(dict(test="basis", venue="kalshi_vs_spot", days=len(prem) / 24, median_ratio=float(ratio.median()),
                        prem_sd_bps=prem.std(), prem_p5_bps=prem.quantile(.05), prem_p95_bps=prem.quantile(.95),
                        abs_gt_20bps_share=(prem.abs() > 20).mean(), ar1=ac, half_life_h=hl,
                        median_spread_bps=j.spread_bps.median()))
        for kb in (10, 20, 40):
            trades, pos, entry = [], 0, 0.0
            for v in prem.values:
                if pos == 0 and abs(v) > kb:
                    pos, entry = -np.sign(v), v                 # short the perp when it is rich, long when cheap
                elif pos != 0 and np.sign(v) != np.sign(entry):  # premium crossed back through zero: close
                    trades.append(pos * (v - entry)); pos = 0     # bps captured (hedged with spot)
            tr = np.array(trades)
            res.append(dict(test="reversion_rule", venue=f"|prem|>{kb}bps, exit at 0", days=len(prem) / 24, trades=len(tr),
                            trades_per_month=len(tr) / (len(prem) / 24 / 30) if len(prem) else np.nan,
                            gross_bps_per_trade=tr.mean() if len(tr) else np.nan,
                            win_share=(tr > 0).mean() if len(tr) else np.nan))
    except Exception as e:
        print(f"::warning::basis test failed: {str(e)[:200]}")
    R = pd.DataFrame(res)
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/perps_edge.csv", index=False)
    print(R.round(3).to_string(index=False))
    fv = R[R.test == "funding_vs_offshore"]
    b = R[R.test == "basis"]
    rv = R[R.test == "reversion_rule"]
    print("::notice::perps edge | funding Kalshi minus venue (ann %, t, share richer): " + "; ".join(
        f"{r.venue} {r.kalshi_ann_pct:+.1f} vs {r.venue_ann_pct:+.1f} -> {r.spread_ann_pct:+.1f} (t {r.t:.1f}, {r.kalshi_richer_share:.0%}, last30 {r.last30_spread_ann_pct:+.1f})"
        for r in fv.itertuples())
        + (f" | basis: sd {b.prem_sd_bps.iloc[0]:.1f}bps, p5/p95 {b.prem_p5_bps.iloc[0]:+.0f}/{b.prem_p95_bps.iloc[0]:+.0f}, half-life {b.half_life_h.iloc[0]:.1f}h, "
           f"spread {b.median_spread_bps.iloc[0]:.1f}bps" if len(b) else " | basis: n/a")
        + (" | reversion: " + "; ".join(f"{r.venue}: {r.trades} trades ({r.trades_per_month:.0f}/mo), {r.gross_bps_per_trade:+.1f}bps gross, {r.win_share:.0%} win"
                                       for r in rv.itertuples()) if len(rv) else ""))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc(); print(tb); print("::error::" + tb[-900:].replace("\n", " | ")); raise
