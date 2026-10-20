"""
research/options_premium.py -- selling SPY option premium with defined risk (Oct 20, 2026).

Question: do short SPY put spreads / call spreads / iron condors, held to expiry, earn money after costs, and which
setup survives a 2024-25 -> 2025-26 split? (Alpaca option history starts Feb 2024, so there is no 2017-23 dev period:
FIRST HALF Feb 2024 - Jun 2025 picks the setup, SECOND HALF Jul 2025 - Oct 2026 checks it.)

Each trading day D, entry at 10:00 ET (15-minute VWAP of each option's trades, 10:00-10:15):
  expiry   0DTE (D), 1DTE (next trading day), ~7DTE (first trading day >= D + 7 calendar days)
  implied move M = ATM call + ATM put (strike = SPY at 10:00 rounded to $1)
  short strike  = spot -/+ k*M rounded to $1, k in {0.5, 1.0, 1.5};  long (insurance) strike = short -/+ W, W in {2, 5}
  structures    put spread, call spread, iron condor (both; risk = W - total credit since only one side can lose)
  costs         $0.03 per leg per share (half-spread ~$0.01-0.02 + fees), both at entry; held to expiry, value =
                intrinsic at the SPY 4pm close of the expiry day (no early exit, no assignment modelled)
  filter        none / skip when prior-day VIX close > 25 (FRED VIXCLS)
Outputs per variant: trades, win %, mean return on risk per trade, worst trade, worst 20-day stretch, and an account
path risking 2% of equity (max loss) per trade: CAGR, max drawdown, Sharpe; plus Aug 5 2024 and the worst days.
Prices are cached in research_data/options/spy_prem_prices.parquet so the analysis can be re-run without refetching.
"""
import datetime as dt, io, itertools, os, time, traceback, urllib.request
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = f"{RD}/options"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{RD}/results", exist_ok=True)
CACHE = f"{OUT}/spy_prem_prices_v2.parquet"   # v1 used adjusted SPY prices (wrong strikes)
START, MID = pd.Timestamp("2024-02-05"), pd.Timestamp("2025-07-01")
KS, WS, COST, RISK_FRAC = (0.5, 1.0, 1.5), (2, 5), 0.03, 0.02


def occ(exp, cp, k):
    return f"SPY{exp:%y%m%d}{cp}{int(round(k * 1000)):08d}"


def spy_days():
    """RAW (unadjusted) SPY 5-minute bars: option strikes are in raw dollars (the research cache is dividend-adjusted)."""
    p = f"{OUT}/spy_5m_raw.parquet"
    if os.path.exists(p) and os.environ.get("SKIP_FETCH") == "1":
        b = pd.read_parquet(p)
    else:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import Adjustment, DataFeed
        sc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
        b = sc.get_stock_bars(StockBarsRequest(symbol_or_symbols="SPY", timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                              start=dt.datetime(2024, 1, 25), adjustment=Adjustment.RAW, feed=DataFeed.SIP)).df.reset_index()
        b["ts"] = b.timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
        b = b[(b.ts.dt.strftime("%H:%M") >= "09:30") & (b.ts.dt.strftime("%H:%M") < "16:00")][["ts", "open", "close"]]
        b.to_parquet(p, index=False)
    b["d"] = b.ts.dt.normalize(); b["tm"] = b.ts.dt.strftime("%H:%M")
    spot = b[b.tm == "10:00"].set_index("d").open
    close = b.groupby("d").close.last()
    return spot, close


def vix():
    try:
        raw = urllib.request.urlopen("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS", timeout=60).read()
        v = pd.read_csv(io.BytesIO(raw)); v.columns = ["d", "vix"]
        v["d"] = pd.to_datetime(v.d); v["vix"] = pd.to_numeric(v.vix, errors="coerce")
        return v.dropna().set_index("d").vix
    except Exception as e:
        print(f"::warning::VIX fetch failed: {e}"); return pd.Series(dtype=float)


def fetch(spot, close):
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    oc = OptionHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    have = pd.read_parquet(CACHE) if os.path.exists(CACHE) else pd.DataFrame()
    done = set(zip(have.d, have.tenor)) if len(have) else set()
    days = [d for d in spot.index if d >= START]
    if os.environ.get("LIMIT"):
        days = days[:int(os.environ["LIMIT"])]
    tdays = list(close.index)
    rows, last_t, n_req = [], 0.0, 0

    def bars(syms, d):
        nonlocal last_t, n_req
        out = {}
        st = pd.Timestamp(d.date()).tz_localize("America/New_York") + pd.Timedelta("10:00:00")
        for i in range(0, len(syms), 90):
            for tries in range(4):
                wait = 0.32 - (time.time() - last_t)
                if wait > 0: time.sleep(wait)
                last_t = time.time(); n_req += 1
                try:
                    df = oc.get_option_bars(OptionBarsRequest(symbol_or_symbols=syms[i:i + 90],
                                            timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=st.tz_convert("UTC"),
                                            end=(st + pd.Timedelta("15min")).tz_convert("UTC"))).df
                    break
                except Exception as e:
                    if tries == 3: print(f"::warning::bars {d.date()} {e}"); df = pd.DataFrame()
                    time.sleep(3 * (tries + 1))
            if len(df):
                df = df.reset_index()
                for s, g in df.groupby("symbol"):
                    v = (g.vwap * g.volume).sum() / g.volume.sum() if g.volume.sum() > 0 else g.close.mean()
                    out[s] = (float(v), int(g.trade_count.sum()) if "trade_count" in g else int(len(g)))
        return out
    for j, d in enumerate(days):
        i = tdays.index(d)
        nxt = [x for x in tdays[i:] if x >= d + pd.Timedelta(days=7)]
        tenors = {"0DTE": d, "1DTE": tdays[i + 1] if i + 1 < len(tdays) else None, "7DTE": nxt[0] if nxt else None}
        s0 = float(spot[d])
        for ten, e in tenors.items():
            if e is None or (d, ten) in done:
                continue
            atm = round(s0)
            b1 = bars([occ(e, "C", atm), occ(e, "P", atm)], d)
            c, p = b1.get(occ(e, "C", atm)), b1.get(occ(e, "P", atm))
            if not c or not p:
                rows.append(dict(d=d, tenor=ten, exp=e, spot=s0, M=np.nan, sym=None, cp=None, k=np.nan, px=np.nan, n=0)); continue
            M = c[0] + p[0]
            need = set()
            for kk, W in itertools.product(KS, WS):
                ps, cs = round(s0 - kk * M), round(s0 + kk * M)
                need |= {("P", ps), ("P", ps - W), ("C", cs), ("C", cs + W)}
            b2 = bars([occ(e, cp, k) for cp, k in sorted(need)], d)
            rows.append(dict(d=d, tenor=ten, exp=e, spot=s0, M=M, sym="ATM", cp="CP", k=atm, px=M, n=c[1] + p[1]))
            for cp, k in need:
                v = b2.get(occ(e, cp, k))
                rows.append(dict(d=d, tenor=ten, exp=e, spot=s0, M=M, sym=occ(e, cp, k), cp=cp, k=k,
                                 px=v[0] if v else np.nan, n=v[1] if v else 0))
        if j % 40 == 0 and rows:
            pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_parquet(CACHE, index=False)
            print(f"{d.date()} {j}/{len(days)} days, {n_req} requests", flush=True)
    allp = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    allp.to_parquet(CACHE, index=False)
    return allp


def trades(P, close, vx, cost=COST):
    out = []
    for (d, ten), g in P[P.M.notna()].groupby(["d", "tenor"]):
        s0, M, e = g.spot.iloc[0], g.M.iloc[0], g.exp.iloc[0]
        if e not in close.index:
            continue
        sT = float(close[e])
        px = {(r.cp, r.k): r.px for r in g.itertuples() if r.cp in ("P", "C")}
        for kk, W in itertools.product(KS, WS):
            ps, cs = round(s0 - kk * M), round(s0 + kk * M)
            legs = {}
            for side, (sk, lk) in (("put", (ps, ps - W)), ("call", (cs, cs + W))):
                a, b = px.get((side[0].upper(), sk)), px.get((side[0].upper(), lk))
                if a is None or b is None or not np.isfinite(a):
                    continue
                if not np.isfinite(b):
                    b = 0.01                                   # untraded far wing: assume a penny
                credit = a - b - 2 * cost
                if side == "put":
                    loss = min(max(sk - sT, 0), W)
                else:
                    loss = min(max(sT - sk, 0), W)
                legs[side] = (credit, loss)
            for struct, sides in (("put spread", ["put"]), ("call spread", ["call"]), ("iron condor", ["put", "call"])):
                if not all(s in legs for s in sides):
                    continue
                cr = sum(legs[s][0] for s in sides); ls = sum(legs[s][1] for s in sides)
                risk = W - cr
                if cr <= 0 or risk <= 0:
                    continue
                out.append(dict(cost=cost, d=d, tenor=ten, struct=struct, k=kk, W=W, credit=cr, pnl=cr - ls, risk=risk,
                                ror=(cr - ls) / risk, vix=vx.asof(d - pd.Timedelta(days=1)) if len(vx) else np.nan,
                                move=sT / s0 - 1, M_pct=M / s0))
    return pd.DataFrame(out)


def summarise(T):
    rows = []
    for key, g in T.groupby(["tenor", "struct", "k", "W", "cost"]):
        for filt in ("none", "VIX<=25"):
            x = g if filt == "none" else g[~(g.vix > 25)]
            for half, m in (("first", x.d < MID), ("second", x.d >= MID), ("all", x.d >= START)):
                y = x[m].sort_values("d")
                if len(y) < 40:
                    continue
                # 7DTE trades overlap: one new trade per day, each risking RISK_FRAC/hold so total risk stays ~2%
                hold = {"0DTE": 1, "1DTE": 1, "7DTE": 5}[key[0]]
                f = RISK_FRAC / hold
                daily = y.groupby(y.d.values).ror.mean() * f
                eq = (1 + daily).cumprod()
                yrs = (y.d.max() - y.d.min()).days / 365.25
                r20 = daily.rolling(20).sum().min()
                rows.append(dict(tenor=key[0], struct=key[1], k=key[2], W=key[3], cost=key[4], filter=filt, half=half, trades=len(y),
                                 win_pct=(y.pnl > 0).mean() * 100, ror_mean_pct=y.ror.mean() * 100,
                                 t=y.ror.mean() / y.ror.std() * np.sqrt(len(y) / hold),
                                 credit_avg=y.credit.mean(), worst_ror_pct=y.ror.min() * 100,
                                 cagr_pct=(eq.iloc[-1] ** (1 / max(yrs, 0.1)) - 1) * 100,
                                 maxdd_pct=((eq / eq.cummax()) - 1).min() * 100, worst20d_pct=r20 * 100,
                                 sharpe=daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan))
    return pd.DataFrame(rows)


def main():
    spot, close = spy_days()
    vx = vix()
    P = fetch(spot, close) if os.environ.get("SKIP_FETCH") != "1" else pd.read_parquet(CACHE)
    ok = P[P.M.notna()]
    print(f"::notice::option prices: {ok[['d','tenor']].drop_duplicates().shape[0]} day-tenors with an ATM price, "
          f"{ok.px.notna().mean():.0%} of legs priced; {P[P.M.isna()][['d','tenor']].drop_duplicates().shape[0]} day-tenors missing ATM")
    T = pd.concat([trades(P, close, vx, c) for c in (0.01, 0.03)], ignore_index=True)
    T.to_parquet(f"{RD}/results/options_premium_trades.parquet", index=False)
    S = summarise(T)
    S.to_csv(f"{RD}/results/options_premium_summary.csv", index=False)
    if not len(S):
        print(f"::notice::premium: {len(T)} trades, too few for a summary"); return
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    W = S[S.half != "all"].pivot_table(index=["tenor", "struct", "k", "W", "cost", "filter"], columns="half",
                                       values=["ror_mean_pct", "t", "win_pct", "sharpe", "maxdd_pct"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    print(W.round(2).sort_values("sharpe_first", ascending=False).to_string())
    for c in (0.01, 0.03):
      f = S[(S.half == "first") & (S.trades >= 100) & (S.cost == c)].sort_values("sharpe", ascending=False).iloc[0]
      s = S[(S.half == "second") & (S.cost == c) & (S.tenor == f.tenor) & (S.struct == f.struct) & (S.k == f.k) & (S.W == f.W) & (S["filter"] == f["filter"])]
      s = s.iloc[0] if len(s) else None
      print(f"::notice::premium cost {c}: first-half pick [{f.tenor} {f.struct} k{f.k} W{f.W} {f['filter']}] Sharpe {f.sharpe:.2f}, "
          f"RoR {f.ror_mean_pct:+.1f}%/trade, win {f.win_pct:.0f}% -> second half "
          + (f"Sharpe {s.sharpe:.2f}, RoR {s.ror_mean_pct:+.1f}%/trade (t {s.t:.1f}), win {s.win_pct:.0f}%, "
             f"CAGR {s.cagr_pct:+.1f}% maxDD {s.maxdd_pct:.1f}% at 2% risk/trade" if s is not None else "n/a"))
    a = T[(T.d == pd.Timestamp("2024-08-05")) & (T.cost == 0.01)]
    if len(a):
        print("Aug 5 2024 trades:\n" + a.groupby(["tenor", "struct", "k", "W"]).ror.mean().mul(100).round(0).to_string())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
