"""
research/intraday_strats.py -- can an active intraday strategy beat its costs? (Oct 19, 2026)

Data: Alpaca SIP 5-minute bars (split/dividend adjusted), 2017-01 .. today, regular session, for
  index ETFs SPY QQQ IWM DIA; 10 sector SPDRs; 48 large liquid stocks (no CAT / dealer-adjacent / machinery names).
Bars are stamped by START time (09:30 bar = 09:30-09:35). Half days (no 15:55 bar) are dropped.

  A  INTRADAY MOMENTUM (Gao-Han-Li-Zhou 2018): sign of the first half hour (prev close -> 10:00, or open -> 10:00),
     optionally plus the 15:00-15:30 half hour, decides a long/short held 15:30 -> close. Index ETFs. One trade/day.
     Variants: signal definition x trade only when |signal| is above a dev-set quantile (0, 50, 75%).
  B  SHORT-TERM REVERSAL: a stock whose L-minute return minus SPY's is below -X is bought at the next bar's open,
     SPY-hedged, held H minutes (exit at bar close). Large stocks + sector ETFs. L {15, 30}, X {0.75, 1, 1.5%}, H {15, 30, 60}.
  C  OPENING-RANGE BREAKOUT: first K minutes' high/low (K 15, 30); first close beyond it -> enter at next open in that
     direction, stop at the other side of the range, else exit at the close. Long-only and long+short. All symbols.
Costs per side: index ETFs 1 bp, sector ETFs and stocks 2 bp (+1 bp per side for the SPY hedge in B).
Each family: the variant with the best dev (2017-2023) daily Sharpe (>= 250 trades) is scored once on 2024-2026.
Reports gross and net per-trade bp, trades/day, daily Sharpe (252), and the break-even cost per side.
"""
import os, itertools
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = f"{RD}/intraday_liquid"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{RD}/results", exist_ok=True)
SPLIT = pd.Timestamp("2024-01-01")
INDEX = ["SPY", "QQQ", "IWM", "DIA"]
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE"]
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V", "MA", "UNH", "XOM", "JNJ", "PG",
          "HD", "COST", "ABBV", "MRK", "PEP", "KO", "BAC", "WMT", "CVX", "LLY", "ORCL", "CSCO", "ADBE", "CRM", "NFLX",
          "AMD", "INTC", "QCOM", "TXN", "MCD", "DIS", "NKE", "T", "VZ", "PFE", "TMO", "ABT", "WFC", "C", "GS", "MS",
          "SBUX", "AMGN"]
COST = {**{s: 1e-4 for s in INDEX}, **{s: 2e-4 for s in SECTORS + STOCKS}}


def fetch(sym):
    p = f"{OUT}/5m_{sym}.parquet"
    if os.path.exists(p):
        return pd.read_parquet(p)
    import datetime as dt
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment, DataFeed
    cl = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    parts = []
    for y in range(2017, dt.date.today().year + 1):
        end = min(dt.datetime(y + 1, 1, 1), dt.datetime.utcnow() - dt.timedelta(minutes=30))
        for k in range(4):
            try:
                d = cl.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                                                       start=dt.datetime(y, 1, 1), end=end, adjustment=Adjustment.ALL,
                                                       feed=DataFeed.SIP)).df.reset_index()
                parts.append(d); break
            except Exception as e:
                print(f"  {sym} {y}: {e}", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d["ts"] = pd.to_datetime(d.timestamp).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d = d[(d.ts.dt.time >= pd.Timestamp("09:30").time()) & (d.ts.dt.time <= pd.Timestamp("15:55").time())]
    d = d[["ts", "open", "high", "low", "close"]].astype({c: "float32" for c in ("open", "high", "low", "close")})
    d.to_parquet(p, index=False)
    return d


def load():
    O, H, L, C = {}, {}, {}, {}
    for s in INDEX + SECTORS + STOCKS:
        d = fetch(s).set_index("ts")
        O[s], H[s], L[s], C[s] = d.open, d.high, d.low, d.close
        print(f"  {s}: {len(d):,} bars", flush=True)
    O, H, L, C = (pd.DataFrame(x).sort_index() for x in (O, H, L, C))
    day = pd.Series(O.index.normalize(), index=O.index)
    full = day.groupby(day).apply(lambda g: (g.index.time == pd.Timestamp("15:55").time()).any())
    keep = day.map(full).values
    return O[keep], H[keep], L[keep], C[keep]


def stats(daily, ntr, gross_bp, net_bp):
    r = daily.dropna()
    if len(r) < 50:
        return dict(days=len(r))
    eq = (1 + r).cumprod()
    return dict(trades=int(ntr), days=len(r), trades_per_day=ntr / len(r), gross_bp=gross_bp, net_bp=net_bp,
                breakeven_cost_bp=gross_bp / 2, ann_pct=(eq.iloc[-1] ** (252 / len(r)) - 1) * 100,
                sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan, maxdd_pct=(eq / eq.cummax() - 1).min() * 100)


def split_stats(tr, family, variant):
    """tr: DataFrame(date, sym, gross, net) one row per trade; daily = equal-weight mean of that day's trades."""
    rows = []
    for per, m in (("dev", tr.date < SPLIT), ("hold", tr.date >= SPLIT)):
        t = tr[m]
        if not len(t):
            continue
        daily = t.groupby("date").net.mean()
        rows.append(dict(family=family, variant=variant, period=per,
                         **stats(daily, len(t), t.gross.mean() * 1e4, t.net.mean() * 1e4)))
    return rows


def at(df, hhmm):
    x = df[df.index.time == pd.Timestamp(hhmm).time()]
    x.index = x.index.normalize()
    return x


def fam_a(O, C):
    rows = []
    c10, c1500, c1525, cl, op = at(C, "09:55"), at(C, "14:55"), at(C, "15:25"), at(C, "15:55"), at(O, "09:30")
    prev = cl.shift(1)
    sig_defs = {"prevclose->10:00": c10 / prev - 1, "open->10:00": c10 / op - 1}
    r12 = c1525 / c1500 - 1
    last = cl / c1525 - 1
    for (sn, s), add12, q in itertools.product(sig_defs.items(), (False, True), (0, 50, 75)):
        sig = s + r12 if add12 else s
        recs = []
        for sym in INDEX:
            x = pd.DataFrame({"sig": sig[sym], "ret": last[sym]}).dropna()
            thr = x.sig.abs()[x.index < SPLIT].quantile(q / 100) if q else 0
            x = x[x.sig.abs() > thr]
            g = np.sign(x.sig) * x.ret
            recs.append(pd.DataFrame({"date": x.index, "sym": sym, "gross": g.values, "net": g.values - 2 * COST[sym]}))
        rows += split_stats(pd.concat(recs), "A intraday momentum", f"{sn}{' +15:00-15:30' if add12 else ''} |sig|>q{q}")
    return rows


def fam_b(O, C):
    rows = []
    syms = SECTORS + STOCKS
    date = pd.Series(C.index.normalize(), index=C.index)
    tod = pd.Series(C.index.time, index=C.index)
    ok_t = (tod >= pd.Timestamp("10:00").time())
    for Lm, X, Hm in itertools.product((15, 30), (0.0075, 0.01, 0.015), (15, 30, 60)):
        Lb, Hb = Lm // 5, Hm // 5
        same = (date == date.shift(Lb)) & (date == date.shift(-Hb)) & ok_t
        R = C / C.shift(Lb) - 1
        ex = R[syms].sub(R["SPY"], axis=0)
        entry = O.shift(-1); exitp = C.shift(-Hb)
        tr_ret = (exitp / entry - 1)
        hedge = tr_ret["SPY"]
        recs = []
        for s in syms:
            m = (ex[s] < -X) & same
            idx = ex.index[m.values]
            if not len(idx):
                continue
            # non-overlapping per symbol: drop signals within the holding window of the previous one
            keep, last_t = [], None
            for t in idx:
                if last_t is None or t > last_t:
                    keep.append(t); last_t = t + pd.Timedelta(minutes=Hm)
            keep = pd.DatetimeIndex(keep)
            g = (tr_ret.loc[keep, s] - hedge.loc[keep]).values
            recs.append(pd.DataFrame({"date": keep.normalize(), "sym": s, "gross": g, "net": g - 2 * COST[s] - 2e-4}))
        if recs:
            rows += split_stats(pd.concat(recs).dropna(), "B short-term reversal", f"L{Lm}m X{X*100:.2f}% H{Hm}m")
    return rows


def fam_c(O, H, L, C):
    rows = []
    syms = INDEX + SECTORS + STOCKS
    for K, both in itertools.product((15, 30), (False, True)):
        Kb = K // 5
        recs = []
        for s in syms:
            df = pd.DataFrame({"o": O[s], "h": H[s], "l": L[s], "c": C[s]}).dropna()
            for d, g in df.groupby(df.index.normalize()):
                if len(g) < Kb + 3:
                    continue
                orh, orl = g.h.iloc[:Kb].max(), g.l.iloc[:Kb].min()
                rest = g.iloc[Kb:]
                up = np.where(rest.c.values > orh)[0]; dn = np.where(rest.c.values < orl)[0]
                iu = up[0] if len(up) else 10**9; idn = dn[0] if len(dn) else 10**9
                if min(iu, idn) >= len(rest) - 1:
                    continue
                side = 1 if iu < idn else -1
                if side == -1 and not both:
                    continue
                i = min(iu, idn) + 1
                px_in = rest.o.iloc[i]; stop = orl if side == 1 else orh
                after = rest.iloc[i:]
                hit = np.where(after.l.values <= stop)[0] if side == 1 else np.where(after.h.values >= stop)[0]
                if len(hit):
                    j = hit[0]; bo = after.o.iloc[j]
                    px_out = min(stop, bo) if side == 1 else max(stop, bo)     # gap through the stop fills worse
                else:
                    px_out = after.c.iloc[-1]
                gr = side * (px_out / px_in - 1)
                recs.append((d, s, gr, gr - 2 * COST[s]))
        tr = pd.DataFrame(recs, columns=["date", "sym", "gross", "net"])
        rows += split_stats(tr, "C opening-range breakout", f"OR{K}m {'long+short' if both else 'long only'}")
    return rows


def main():
    O, H, L, C = load()
    print(f"bars: {len(C):,} x {C.shape[1]} symbols, {C.index.normalize().nunique():,} days", flush=True)
    rows = fam_a(O, C)
    print("A done", flush=True)
    rows += fam_b(O, C)
    print("B done", flush=True)
    rows += fam_c(O, H, L, C)
    print("C done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(f"{RD}/results/intraday_strats.csv", index=False)
    W = R.pivot_table(index=["family", "variant"], columns="period",
                      values=["trades_per_day", "gross_bp", "net_bp", "breakeven_cost_bp", "sharpe", "ann_pct", "maxdd_pct"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
    cols = ["trades_per_day_dev", "gross_bp_dev", "net_bp_dev", "sharpe_dev", "gross_bp_hold", "net_bp_hold", "breakeven_cost_bp_hold",
            "sharpe_hold", "ann_pct_hold", "maxdd_pct_hold"]
    print(W[[c for c in cols if c in W]].round(2).to_string())
    for fam in R.family.unique():
        d = R[(R.family == fam) & (R.period == "dev") & (R.trades >= 250)].sort_values("sharpe", ascending=False)
        if not len(d):
            continue
        v = d.iloc[0].variant
        h = R[(R.family == fam) & (R.variant == v) & (R.period == "hold")]
        hs = h.iloc[0] if len(h) else None
        print(f"::notice::{fam}: dev pick [{v}] dev Sharpe {d.iloc[0].sharpe:.2f} net {d.iloc[0].net_bp:+.1f}bp/trade -> "
              + (f"holdout Sharpe {hs.sharpe:.2f}, net {hs.net_bp:+.1f}bp/trade (gross {hs.gross_bp:+.1f}), {hs.ann_pct:+.1f}%/yr, "
                 f"maxDD {hs.maxdd_pct:.1f}%, {hs.trades_per_day:.1f} trades/day" if hs is not None else "no holdout trades"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
