"""
research/probes/alpaca_equity.py -- read-only: the paper account's daily equity since
inception, plus SPY/IWM closes, printed as ::notice:: lines (logs are not readable
here). Weekly table (Fri close to Fri close) and the Apr 17 - May 29, 2026 window.
"""
import datetime as dt
import os

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed

K, S = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
tc = TradingClient(K, S, paper=True)
h = tc.get_portfolio_history(GetPortfolioHistoryRequest(period="1A", timeframe="1D", extended_hours=False))
eq = pd.Series(h.equity, index=pd.to_datetime(h.timestamp, unit="s").tz_localize("UTC").tz_convert("America/New_York").date)
eq = eq[eq > 0]
eq.index = pd.to_datetime(eq.index)
dc = StockHistoricalDataClient(K, S)
b = dc.get_stock_bars(StockBarsRequest(symbol_or_symbols=["SPY", "IWM"], timeframe=TimeFrame.Day,
                                       start=dt.datetime(2026, 3, 1), adjustment=Adjustment.ALL,
                                       feed=DataFeed.SIP)).df.reset_index()
b["date"] = pd.to_datetime(b.timestamp).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
px = b.pivot_table(index="date", columns="symbol", values="close")
df = pd.concat([eq.rename("equity"), px], axis=1).dropna()
w = df.resample("W-FRI").last()
wr = (w / w.shift(1) - 1) * 100
print(f"::notice::equity first {df.index[0].date()} {df.equity.iloc[0]:.0f}, last {df.index[-1].date()} {df.equity.iloc[-1]:.0f}")
rows = [f"{d.date()} eq {w.equity[d]:.0f} wk {wr.equity[d]:+.1f}% SPY {wr.SPY[d]:+.1f}% IWM {wr.IWM[d]:+.1f}%"
        for d in wr.index[1:]]
for i in range(0, len(rows), 8):
    print("::notice::" + " | ".join(rows[i:i + 8]))
win = df.loc["2026-04-17":"2026-05-29"]
r = win.iloc[-1] / win.iloc[0] - 1
print(f"::notice::Apr17-May29: account {r.equity * 100:+.1f}%, SPY {r.SPY * 100:+.1f}%, IWM {r.IWM * 100:+.1f}%")
pk = df.equity.cummax()
print(f"::notice::peak {df.equity.max():.0f} on {df.equity.idxmax().date()}; max DD {(df.equity / pk - 1).min() * 100:.1f}%")
