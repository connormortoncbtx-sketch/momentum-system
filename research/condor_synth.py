"""
research/condor_synth.py -- the same-day condor on 10 years of MODELLED prices (Oct 23, 2026)

Real option prices (Alpaca) start Feb 2024, so the backtest saw one market regime. This rebuilds the trade from free
data back to 2016 to see how it fares in 2018 (Feb vol blow-up), 2020 (crash) and 2022 (bear market):
  spot     SPY / QQQ raw 15-minute bars (Alpaca SIP): 10:00 ET open and the 4pm close
  implied  the 10:00 implied move M modelled from the prior VIX (VXN for QQQ) close:
           log(M / S) = a + b log(VIX), fitted on the REAL 2024-26 implied moves (options_premium trades)
  credit   Black-Scholes-normal with the volatility that makes the ATM straddle = M, the condor at k = 1 (shorts S -/+ M
           rounded to $1, wings SPY $5 / QQQ 0.8%); times a factor fitted so the modelled credit matches the real
           2024-26 credits on average (absorbs skew)
  P&L      credit - $0.01 or $0.03 per leg - loss at the close; return on risk = P&L / (width - net credit)
Validation on 2024-26: modelled vs real return on risk, day by day. Then per year: trades, mean RoR, t, win %, max-loss
days, worst day, and the 2%/day book's return and drawdown -- all weekdays, and Mon/Wed/Fri only (SPY's 0DTE expiries
before 2022). It is a MODEL: real premiums rise in stress (helping), fills widen (hurting); the point is the tail.
"""
import datetime as dt, io, os, urllib.request
import numpy as np, pandas as pd
from scipy.stats import norm

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = f"{RD}/results"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 220)


def fred(series):
    raw = urllib.request.urlopen(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", timeout=60).read()
    v = pd.read_csv(io.BytesIO(raw)); v.columns = ["d", "v"]
    v["d"] = pd.to_datetime(v.d); v["v"] = pd.to_numeric(v.v, errors="coerce")
    return v.dropna().set_index("d").v


def bars(und):
    p = f"{OUT}/{und.lower()}_15m_raw_2016.parquet"
    if os.path.exists(p):
        return pd.read_parquet(p)
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment, DataFeed
    sc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    parts = []
    for y in range(2016, dt.date.today().year + 1):
        b = sc.get_stock_bars(StockBarsRequest(symbol_or_symbols=und, timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                              start=dt.datetime(y, 1, 1), end=min(dt.datetime(y + 1, 1, 1), dt.datetime.utcnow() - dt.timedelta(minutes=30)),
                              adjustment=Adjustment.RAW, feed=DataFeed.SIP)).df.reset_index()
        parts.append(b)
    b = pd.concat(parts)
    b["ts"] = b.timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    b = b[(b.ts.dt.strftime("%H:%M") >= "09:30") & (b.ts.dt.strftime("%H:%M") < "16:00")][["ts", "open", "close"]]
    b.to_parquet(p, index=False)
    return b


def days(und):
    b = bars(und)
    b["d"] = b.ts.dt.normalize(); b["tm"] = b.ts.dt.strftime("%H:%M")
    s0 = b[b.tm == "10:00"].set_index("d").open
    sT = b.groupby("d").close.last()
    return pd.DataFrame({"s0": s0, "sT": sT}).dropna()


def condor_value(S, ps, cs, W, v):
    """Black-Scholes-normal (zero rate) value of short put spread ps/ps-W + short call spread cs/cs+W, sd of log price v."""
    def put(K):
        d1 = (np.log(S / K) + v * v / 2) / v
        return K * norm.cdf(-(d1 - v)) - S * norm.cdf(-d1)
    def call(K):
        d1 = (np.log(S / K) + v * v / 2) / v
        return S * norm.cdf(d1) - K * norm.cdf(d1 - v)
    return put(ps) - put(ps - W) + call(cs) - call(cs + W)


def run(und, volser):
    d = days(und)
    vix = fred(volser)
    d["vix"] = [vix.asof(x - pd.Timedelta(days=1)) for x in d.index]
    real = pd.read_parquet(f"{OUT}/options_premium_trades.parquet" if und == "SPY" else f"{OUT}/options_premium_trades_{und}.parquet")
    Wl = "5" if und == "SPY" else "0.8%"
    real = real[(real.tenor == "0DTE") & (real.struct == "iron condor") & (real.k == 1.0) & (real.W.astype(str) == Wl)]
    r1 = real[real.cost == 0.01].assign(d=lambda x: pd.to_datetime(x.d)).set_index("d")
    # 1. implied move model, fitted on real 2024-26 implied moves
    j = r1[["M_pct"]].join(d[["vix"]], how="inner").dropna()
    b, a = np.polyfit(np.log(j.vix), np.log(j.M_pct), 1)
    d["M"] = d.s0 * np.exp(a + b * np.log(d.vix))
    d["W"] = 5 if und == "SPY" else np.maximum(1, (0.008 * d.s0).round())
    d["ps"] = (d.s0 - d.M).round(); d["cs"] = (d.s0 + d.M).round()
    v = d.M / (0.7979 * d.s0)
    d["credit_raw"] = condor_value(d.s0.values, d.ps.values, d.cs.values, d.W.values, v.values)
    # 2. credit scale: modelled vs real gross credit (credit + 4 x cost) on the overlap
    jj = r1[["credit"]].assign(gross=lambda x: x.credit + 4 * 0.01).join(d[["credit_raw"]], how="inner").dropna()
    fac = float(np.median(jj.gross / jj.credit_raw))
    d["credit"] = d.credit_raw * fac
    loss = np.minimum(np.maximum(d.ps - d.sT, 0), d.W) + np.minimum(np.maximum(d.sT - d.cs, 0), d.W)
    print(f"{und}: implied move log-fit a {a:.3f} b {b:.3f} (fit on {len(j)} days); credit factor {fac:.2f}")
    rows, val = [], []
    for cost in (0.01, 0.03):
        net = d.credit - 4 * cost
        ok = (net > 0.02) & (net < 0.6 * d.W)                     # as in the real backtest: credit >= 60% of width = bad data
        x = d[ok].assign(ror=((net - loss) / (d.W - net))[ok], maxloss=(loss[ok] >= d.W[ok] - 1e-9))
        # validation on the overlap
        rr = real[real.cost == cost].assign(d=lambda z: pd.to_datetime(z.d)).set_index("d").ror
        jv = x[["ror"]].join(rr.rename("real"), how="inner").dropna()
        val.append(dict(und=und, cost=cost, days=len(jv), corr=jv.ror.corr(jv.real), model_mean_pct=jv.ror.mean() * 100,
                        real_mean_pct=jv.real.mean() * 100))
        for mwf in (False, True):
            y = x[x.index.dayofweek.isin([0, 2, 4])] if mwf else x
            for yr, g in list(y.groupby(y.index.year)) + [("ALL", y), ("2016-23", y[y.index.year <= 2023]), ("2024-26", y[y.index.year >= 2024])]:
                if len(g) < 20:
                    continue
                nav = (1 + 0.02 * g.ror).cumprod()
                rows.append(dict(und=und, cost=cost, days="MWF" if mwf else "all", year=yr, trades=len(g),
                                 ror_mean_pct=g.ror.mean() * 100, t=g.ror.mean() / g.ror.std() * np.sqrt(len(g)),
                                 win_pct=(g.ror > 0).mean() * 100, maxloss_pct=g.maxloss.mean() * 100,
                                 worst_pct=g.ror.min() * 100, book_2pct_ret_pct=(nav.iloc[-1] - 1) * 100,
                                 book_2pct_dd_pct=(nav / nav.cummax() - 1).min() * 100))
    return pd.DataFrame(rows), pd.DataFrame(val)


def main():
    R, V = [], []
    for und, vs in (("SPY", "VIXCLS"), ("QQQ", "VXNCLS")):
        try:
            r, v = run(und, vs); R.append(r); V.append(v)
        except Exception as e:
            print(f"::warning::{und}: {e}")
    R, V = pd.concat(R), pd.concat(V)
    print("\nVALIDATION 2024-26 (modelled vs real return on risk)\n" + V.round(3).to_string(index=False))
    print("\nBY YEAR\n" + R.round(2).to_string(index=False))
    R.to_csv(f"{OUT}/condor_synth.csv", index=False); V.to_csv(f"{OUT}/condor_synth_validation.csv", index=False)
    s = R[(R.days == "all") & (R.year.astype(str).isin(["2016-23", "2024-26", "2018", "2020", "2022"]))]
    print("::notice::condor synth (RoR mean % / book 2%/day return % / DD %): " + "; ".join(
        f"{r.und} {r.cost} {r.year}: {r.ror_mean_pct:+.1f} / {r.book_2pct_ret_pct:+.0f} / {r.book_2pct_dd_pct:.0f}" for r in s.itertuples())
        + " | validation corr " + ", ".join(f"{r.und} {r.cost}: {r.corr:.2f} (model {r.model_mean_pct:+.1f} vs real {r.real_mean_pct:+.1f})" for r in V.itertuples()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc(); print(tb); print("::error::" + tb[-900:].replace("\n", " | ")); raise
