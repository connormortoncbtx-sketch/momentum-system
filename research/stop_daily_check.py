"""
research/stop_daily_check.py -- validate the slow-momentum 10% weekly stop on DAILY bars (Oct 16, 2026).

research/avoid_stack.py approximates the stop with the week's low after Monday and fills
at exactly -10%. Here, for every (formation week, holding) of the SLOW book 2017-2026:
entry = Monday close after formation; walk Tue .. next Monday on daily Alpaca bars
(SIP, split/dividend adjusted): a day OPENING below the stop fills at the open (gap risk);
otherwise a LOW below the stop fills at the stop; else exit at next Monday's close.
Writes research_data/results/stop_daily.parquet (week, symbol, r_stop, r_nostop) and
compares: no stop / weekly approximation / daily-validated.
"""
import os, sys, datetime as dt
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import avoid_stack as A

RD = os.environ.get("PANEL", "research_data")
A.HOLDLOG.clear()
base = A.sim("SLOW", [], log=True)
H = pd.DataFrame(A.HOLDLOG, columns=["week", "symbol"]).drop_duplicates()
print(f"holdings: {len(H):,} name-weeks, {H.symbol.nunique():,} symbols", flush=True)

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed
cl = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
frames, syms = [], sorted(H.symbol.unique())
for i in range(0, len(syms), 100):
    try:
        d = cl.get_stock_bars(StockBarsRequest(symbol_or_symbols=syms[i:i + 100], timeframe=TimeFrame.Day,
                                               start=dt.datetime(2016, 6, 1), adjustment=Adjustment.ALL,
                                               feed=DataFeed.SIP)).df
        if len(d):
            frames.append(d.reset_index()[["symbol", "timestamp", "open", "high", "low", "close"]])
    except Exception as e:
        print(f"bars batch {i}: {e}")
b = pd.concat(frames)
b["date"] = pd.to_datetime(b.timestamp).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
G = {s: g.set_index("date").sort_index() for s, g in b.groupby("symbol")}

STOP, COST = 0.10, A.COST / 1e4
rows = []
for wk, sym in zip(H.week, H.symbol):
    g = G.get(sym)
    if g is None:
        continue
    after = g[g.index > wk]
    if len(after) < 2:
        continue
    mon0 = after.index[0]                                   # first session after formation Friday (Mon, or Tue on holidays)
    nxt = after[after.index >= mon0 + pd.Timedelta(days=7 - mon0.weekday())]   # sessions from the next week
    if nxt.empty:
        continue
    mon1 = nxt.index[0]
    e = after.close.iloc[0]
    path = after[(after.index > mon0) & (after.index <= mon1)]
    s_px = e * (1 - STOP)
    r_stop, how = None, "none"
    for d, row in path.iterrows():
        if row.open <= s_px:
            r_stop, how = row.open / e - 1 - COST, "gap"; break
        if row.low <= s_px:
            r_stop, how = -STOP - COST, "intraday"; break
    r_no = path.close.iloc[-1] / e - 1 if len(path) else np.nan
    rows.append({"week": wk, "symbol": sym, "r_nostop": r_no, "r_stop": r_no if r_stop is None else r_stop, "how": how})
R = pd.DataFrame(rows)
os.makedirs(f"{RD}/results", exist_ok=True)
R.to_parquet(f"{RD}/results/stop_daily.parquet", index=False)
hit = R[R.how != "none"]
print(f"\nname-weeks checked {len(R):,}; stopped {len(hit):,} ({len(hit) / len(R) * 100:.1f}%): "
      f"intraday {int((R.how == 'intraday').sum())}, gap {int((R.how == 'gap').sum())}")
print(f"mean fill on stopped weeks: {hit.r_stop.mean() * 100:.2f}% (gap fills average {R[R.how == 'gap'].r_stop.mean() * 100:.2f}%, worst {R.r_stop.min() * 100:.1f}%)")
print(f"those weeks without the stop would have returned {hit.r_nostop.mean() * 100:.2f}% on average")

os.environ["STOP_DAILY"] = f"{RD}/results/stop_daily.parquet"
A.STOP_OVR = {(pd.Timestamp(w), s): r for w, s, r in zip(R.week, R.symbol, R.r_stop)}
approx_ovr = A.STOP_OVR
A.STOP_OVR = {}
r_apx = A.sim("SLOW", ["stop10"])
A.STOP_OVR = approx_ovr
r_day = A.sim("SLOW", ["stop10"])
for nm, r in (("no stop", base), ("stop, weekly-low approximation", r_apx), ("stop, daily-validated", r_day)):
    d, h = A.both(r)
    print(f"  {nm:32s} dev {A.fmt(d)} | holdout {A.fmt(h)}")
print(f"::notice::stop check: stopped {len(hit) / len(R) * 100:.1f}% of name-weeks, gap fills {int((R.how == 'gap').sum())}; "
      f"daily-validated SLOW+stop dev Sharpe {A.both(r_day)[0]['Sharpe']:.2f} holdout {A.both(r_day)[1]['Sharpe']:.2f} "
      f"(approx {A.both(r_apx)[0]['Sharpe']:.2f}/{A.both(r_apx)[1]['Sharpe']:.2f}, no stop {A.both(base)[0]['Sharpe']:.2f}/{A.both(base)[1]['Sharpe']:.2f})")
