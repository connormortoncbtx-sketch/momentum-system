"""
research/overnight_momentum.py -- hold momentum stocks ONLY overnight (Oct 17, 2026).

Found by following a "bug that helped": the live system's mistimed entries (Tue open in, Mon open
out) beat the designed timing. On 5-minute data the proxy's top picks earn +12/+18 bp per NIGHT
(t 9.4 / 4.7, dev / holdout) and lose -12/-18 bp per DAY (t -5.7 / -4.4); IWM-hedged nights are
still +7 / +11.5 bp (t 4.2 / 2.5). This is the full test:

  universe   weekly research universe (price >= $5, >= $2M/day), names in it at the formation Friday
  signals    LIVE proxy composite, 12-1 momentum, last-week return (hot), chosen at the Friday close
  holding    buy at the CLOSE (closing auction, MOC), sell at the next OPEN (opening auction, MOO),
             every night from the Monday close after formation to the next formation's Monday close;
             variants with / without the weekend night (Fri close -> Mon open)
  prices     Alpaca SIP daily bars (split/dividend adjusted): official open and close prints
  costs      0 / 1 / 2 / 5 bp per side (auctions: no spread; fees ~0.3 bp; impact at small size)
  hedge      minus beta x IWM's own overnight return (beta fitted on dev)
Dev 2017-2023 picks the design; holdout 2024-2026 scored once. Also reports the DAYTIME return of
the same baskets (the other half of the tug of war).
"""
import contextlib, io, itertools, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

RD = os.environ.get("RESEARCH_DATA", os.environ.get("PANEL", "research_data"))
os.environ.setdefault("PANEL", RD)
OUT = f"{RD}/overnight"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{RD}/results", exist_ok=True)
SPLIT, START = pd.Timestamp("2024-01-01"), pd.Timestamp("2017-01-06")

p = B.load_panel(RD, "2026-10-02"); f = B.features(p)
U = B.universe(p, f, 5.0, 2e6)
rk = lambda x: B.xrank(x, U)
SCORES = {"LIVE": (0.35 * rk(f["rs_live"]) + 0.25 * rk(f["trend"]) + 0.20 * rk(f["hi52"])) / 0.80,
          "M12_1": f["m12_1"], "HOT1W": f["r1w"]}
TOPN = (10, 20, 50)
# symbols ever in a top-50 of any score -> daily bars needed only for those
need = set(["IWM", "SPY"])
for s in SCORES.values():
    r = s.where(U).rank(axis=1, ascending=False)
    need |= set(r.columns[(r <= 50).any(axis=0)])
print(f"symbols needing daily bars: {len(need):,}", flush=True)


def bars(symbols):
    fpath = f"{OUT}/daily_oc.parquet"
    if os.path.exists(fpath):
        return pd.read_parquet(fpath)
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    import datetime as dt
    cl = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    syms, parts = sorted(symbols), []
    for i in range(0, len(syms), 100):
        for k in range(4):
            try:
                d = cl.get_stock_bars(StockBarsRequest(symbol_or_symbols=syms[i:i + 100], timeframe=TimeFrame.Day,
                                                       start=dt.datetime(2016, 6, 1), adjustment=Adjustment.ALL,
                                                       feed=DataFeed.SIP)).df.reset_index()
                parts.append(d[["symbol", "timestamp", "open", "close"]]); break
            except Exception as e:
                print(f"  bars {i}: {e}", flush=True)
        if i % 1000 == 0:
            print(f"  bars {i:,}/{len(syms):,}", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d["date"] = pd.to_datetime(d.timestamp).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    d = d.drop(columns="timestamp"); d[["open", "close"]] = d[["open", "close"]].astype("float32")
    d.to_parquet(fpath, index=False)
    return d


d = bars(need)
O = d.pivot_table(index="date", columns="symbol", values="open").sort_index()
C = d.pivot_table(index="date", columns="symbol", values="close").sort_index().reindex_like(O)
night = (O / C.shift(1) - 1).clip(-0.6, 1.5)          # close(t-1) -> open(t): earned on day t's open
day = (C / O - 1).clip(-0.6, 1.5)                     # open(t) -> close(t)
days = O.index
is_mon_open = pd.Series(days.weekday == 0, index=days)   # night that spans the weekend (or a holiday Mon)
weekend = pd.Series(np.r_[False, (np.diff(days.values).astype("timedelta64[D]").astype(int) > 1)], index=days)

# map each trading day to its active formation week: picks formed Friday F trade from the Monday close
# after F; the first night earned is Monday close -> Tuesday open.
weeks = p.weeks
first_mon_close = {}
for w in weeks:
    nxt = days[days > w]
    if len(nxt):
        first_mon_close[w] = nxt[0]                    # Monday (or first trading day) after formation Friday
active = pd.Series(pd.NaT, index=days)
for w, mc in first_mon_close.items():
    active.loc[days > mc] = w                          # nights earned on opens AFTER the entry close
active = active.ffill()


def picks(score_name, n):
    s = SCORES[score_name].where(U)
    out = {}
    for w in weeks:
        row = s.loc[w].dropna()
        if len(row) >= n * 3:
            out[w] = list(row.nlargest(n).index)
    return out


def run(score_name, n, with_weekend, cost_bp):
    P = picks(score_name, n)
    rn, rd = [], []
    prev = None
    for t in days[days >= START]:
        w = active.get(t)
        if pd.isna(w) or w not in P:
            rn.append(np.nan); rd.append(np.nan); continue
        names = [x for x in P[w] if x in night.columns]
        if weekend[t] and not with_weekend:
            rn.append(0.0)
        else:
            v = night.loc[t, names].dropna()
            rn.append(v.mean() - 2 * cost_bp / 1e4 if len(v) else np.nan)
        dv = day.loc[t, names].dropna()
        rd.append(dv.mean() if len(dv) else np.nan)
    idx = days[days >= START]
    return pd.Series(rn, index=idx).dropna(), pd.Series(rd, index=idx).dropna()


def st(r, per_yr=252):
    if len(r) < 50:
        return dict(ann=np.nan, Sh=np.nan, DD=np.nan)
    eq = (1 + r).cumprod()
    return dict(ann=(eq.iloc[-1] ** (per_yr / len(r)) - 1) * 100, Sh=r.mean() / r.std() * np.sqrt(per_yr),
                DD=(eq / eq.cummax() - 1).min() * 100)


def main():
    global R
    iwm_n = night["IWM"].dropna()
    rows, keep = [], {}
    for sc, n, wk in itertools.product(SCORES, TOPN, (True, False)):
        rn0, rday = run(sc, n, wk, 0.0)
        for cost in (0.0, 1.0, 2.0, 5.0):
            rn = rn0 - (2 * cost / 1e4) * (rn0 != 0)
            dv, hv = rn[rn.index < SPLIT], rn[rn.index >= SPLIT]
            b = np.polyfit(iwm_n.reindex(dv.index).fillna(0), dv, 1)[0]
            hed = rn - b * iwm_n.reindex(rn.index).fillna(0)
            a, h = st(dv), st(hv); ah, hh = st(hed[hed.index < SPLIT]), st(hed[hed.index >= SPLIT])
            rows.append(dict(score=sc, n=n, weekend=wk, cost_bp=cost, dev_Sh=a["Sh"], dev_ann=a["ann"], dev_DD=a["DD"],
                             hold_Sh=h["Sh"], hold_ann=h["ann"], hold_DD=h["DD"], beta=b, dev_hSh=ah["Sh"], hold_hSh=hh["Sh"],
                             day_bp_dev=rday[rday.index < SPLIT].mean() * 1e4, day_bp_hold=rday[rday.index >= SPLIT].mean() * 1e4,
                             night_bp_dev=rn0[rn0.index < SPLIT].mean() * 1e4, night_bp_hold=rn0[rn0.index >= SPLIT].mean() * 1e4))
            keep[(sc, n, wk, cost)] = rn
        print(f"  {sc} n{n} weekend={wk} done", flush=True)
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print("\nRAW NIGHT / DAY returns of the baskets (bp, before costs)")
    print(R[R.cost_bp == 0][["score", "n", "weekend", "night_bp_dev", "night_bp_hold", "day_bp_dev", "day_bp_hold"]].round(1).to_string(index=False))
    print("\nOVERNIGHT BOOK by cost (Sharpe on daily returns, annualised 252; hSh = IWM-overnight-hedged)")
    print(R[["score", "n", "weekend", "cost_bp", "dev_Sh", "dev_ann", "dev_DD", "hold_Sh", "hold_ann", "hold_DD", "dev_hSh", "hold_hSh", "beta"]]
          .round(2).to_string(index=False))
    R.to_csv(f"{RD}/results/overnight_momentum.csv", index=False)
    for cost in (1.0, 2.0):
        best = R[R.cost_bp == cost].sort_values("dev_hSh", ascending=False).iloc[0]
        print(f"::notice::overnight momentum @ {cost:.0f}bp/side, dev pick {best.score} n{best.n} weekend={best.weekend}: "
              f"hedged Sharpe dev {best.dev_hSh:.2f} / holdout {best.hold_hSh:.2f}; raw {best.dev_ann:.0f}%/yr / {best.hold_ann:.0f}%/yr")
        pd.DataFrame({"overnight": keep[(best.score, best.n, best.weekend, cost)]}).to_csv(
            f"{RD}/results/overnight_daily_{int(cost)}bp.csv")


if __name__ == "__main__":
    main()
