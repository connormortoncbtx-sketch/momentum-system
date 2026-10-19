"""Re-score the dev-picked MTF variant (60m>0.4%, dip15>0.3%, hold to close, long only, no daily filter) with
per-symbol costs: entry = market order at the next bar open, paying half of a 1-cent spread (+0.2 bp slippage);
exit = closing-auction order (no spread, +0.2 bp). Same signals, same dev/holdout split -- costs only."""
import os, numpy as np, pandas as pd
import intraday_strats as S
O, H, L, C = S.load()
date = pd.Series(C.index.normalize(), index=C.index)
tod = pd.Series(C.index.time, index=C.index)
win = (tod >= pd.Timestamp("10:30").time()) & (tod <= pd.Timestamp("15:00").time())
tr60 = C / C.shift(12) - 1; dip15 = C.shift(1) / C.shift(4) - 1; up = C > C.shift(1)
mask = (win & (date == date.shift(12))).astype(bool)
sig = ((tr60 > 0.004) & (dip15 < -0.003) & up).fillna(False).astype(bool).mul(mask, axis=0).astype(bool)
entry = O.shift(-1); last_c = C.groupby(C.index.normalize()).transform("last")
st = sig.stack(); st = st[st]
df = st.reset_index(); df.columns = ["ts", "sym", "_"]; df["date"] = df.ts.dt.normalize()
df = df.drop_duplicates(["sym", "date"])
ix = pd.MultiIndex.from_arrays([df.ts, df.sym])
df["px_in"] = entry.stack().reindex(ix).values; df["px_out"] = last_c.stack().reindex(ix).values
df = df.dropna(subset=["px_in", "px_out"])
df["gross"] = df.px_out / df.px_in - 1
df["cost"] = 0.005 / df.px_in + 0.2e-4 + 0.2e-4                    # half of a 1c spread on entry + slippage both legs
df["net"] = df.gross - df.cost
df["hold"] = df.date >= pd.Timestamp("2024-01-01")
df["pb"] = pd.cut(df.px_in, [0, 30, 60, 120, 250, 1e9], labels=["<30", "30-60", "60-120", "120-250", ">250"])
pd.set_option("display.width", 200)
def f(g):
    d = g.groupby("date").net.mean()
    return pd.Series(dict(trades=len(g), per_day=len(g) / g.date.nunique(), cost_bp=g.cost.mean() * 1e4, gross_bp=g.gross.mean() * 1e4,
                          net_bp=g.net.mean() * 1e4, t=g.net.mean() / (g.groupby("date").net.mean().std() / np.sqrt(g.date.nunique())),
                          day_sharpe=d.mean() / d.std() * np.sqrt(252), ann_pct=((1 + d).prod() ** (252 / len(d)) - 1) * 100))
print(df.groupby("hold").apply(f).round(2).to_string())
print(df.groupby(["hold", "pb"], observed=True).apply(f).round(2).to_string())
