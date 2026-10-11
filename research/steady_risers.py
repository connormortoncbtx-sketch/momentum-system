"""
research/steady_risers.py -- "steady compounders" at the 3%/week run rate (Oct 2026)

Run rate: 3%/wk = +12.6% per 4 weeks = +46.9% per 13 weeks.
A. How common are steady runs? 13-week windows with return >= +46.9%, at least
   9 of 13 up weeks, no single week > +20%, max drawdown inside the window <= 12%.
B. Can they be caught early? At week t, among stocks that have been rising
   steadily for 8 weeks (+20%+, >= 6 up weeks, no week > +15%) vs stocks with the
   same 8-week gain made in jumps: forward 4/13-week returns and how often the
   NEXT 13 weeks hit the run rate. Dev < 2023, holdout 2023+.
C. "Frog in the pan" (smooth vs jumpy momentum): top 10% by 26-week return, split
   by how many of those weeks were up; forward returns.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

P = os.environ.get("PANEL", "research_data")
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
C, d1c = p.df("close"), p.df("d1_close")
U = (f["price"] >= 2) & (f["adv"] >= 5e5) & (f["hist"] >= 26) & ~pd.Series(p.is_fund, index=p.symbols)
r = (C / C.shift(1) - 1).where(lambda x: x.abs() < 3)
up = (r > 0).astype(float).where(r.notna())
start, split = pd.Timestamp("2017-01-06"), pd.Timestamp("2023-01-01")
RR4, RR13 = 1.03 ** 4 - 1, 1.03 ** 13 - 1

# A. steady 13-week windows (ending at week t)
ret13 = C / C.shift(13) - 1
ups13 = up.rolling(13, min_periods=13).sum()
maxwk13 = r.rolling(13, min_periods=13).max()
roll_max = C.rolling(13, min_periods=13).max()
dd13 = (C / C.rolling(13, min_periods=13).max() - 1).rolling(13, min_periods=13).min()
steady = (ret13 >= RR13) & (ups13 >= 9) & (maxwk13 <= 0.20) & (dd13 >= -0.12) & U.shift(13, fill_value=False)
anyrun = (ret13 >= RR13) & U.shift(13, fill_value=False)
s = steady.loc[start:]; a = anyrun.loc[start:]
first = s & ~s.shift(1, fill_value=False)           # episode starts
print("A. 13-week windows at >= the 3%/wk run rate (+46.9%)")
print(f"   any path: {int(a.sum().sum()):,} stock-weeks; steady path: {int(s.sum().sum()):,} stock-weeks, "
      f"{int(first.sum().sum()):,} distinct steady episodes")
yr = pd.DataFrame({"steady_episodes": first.sum(axis=1).groupby(first.index.year).sum(),
                   "universe_avg": U.loc[start:].sum(axis=1).groupby(U.loc[start:].index.year).mean().round(0)})
print(yr.to_string())
share = (s.sum(axis=1) / U.loc[start:].sum(axis=1)).mean() * 100
print(f"   on an average week, {share:.2f}% of the universe is in a steady 13-week run")

# B. caught early?
ret8 = C / C.shift(8) - 1
ups8 = up.rolling(8, min_periods=8).sum()
max8 = r.rolling(8, min_periods=8).max()
fwd = {h: (C.shift(-h - 1) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 5) for h in (4, 13)}   # Mon close -> Fri close h weeks on
uni_fwd = {h: v.where(U).mean(axis=1) for h, v in fwd.items()}
groups = {
    "steady 8wk riser (+20%, >=6 up wks, no wk >15%)": (ret8 >= 0.20) & (ups8 >= 6) & (max8 <= 0.15),
    "jumpy 8wk riser (+20%, some wk >25%)": (ret8 >= 0.20) & (max8 > 0.25),
    "steady, stronger (+30%, >=6 up, no wk >15%)": (ret8 >= 0.30) & (ups8 >= 6) & (max8 <= 0.15),
    "all stocks": pd.DataFrame(True, index=C.index, columns=C.columns),
}
rows = []
for name, g in groups.items():
    g = g & U
    for per, sl in (("dev", (C.index >= start) & (C.index < split)), ("holdout", C.index >= split)):
        row = {"group": name, "period": per}
        for h in (4, 13):
            x = fwd[h].where(g)[sl]
            exs = x.sub(uni_fwd[h][sl], axis=0)
            row[f"n"] = int(x.notna().sum().sum())
            row[f"fwd{h}_median%"] = np.nanmedian(x.to_numpy()) * 100
            row[f"fwd{h}_excess_mean%"] = np.nanmean(exs.to_numpy()) * 100
        x13 = fwd[13].where(g)[sl].to_numpy().ravel()
        x13 = x13[np.isfinite(x13)]
        row["P(next13 >= run rate)%"] = (x13 >= RR13).mean() * 100
        row["P(next13 < -20%)%"] = (x13 < -0.20).mean() * 100
        rows.append(row)
print("\nB. after 8 weeks of rising: what the NEXT 4 and 13 weeks look like")
print(pd.DataFrame(rows).round(2).to_string(index=False))

# C. frog in the pan within the top decile of 26-week return
ret26 = C / C.shift(26) - 1
ups26 = up.rolling(26, min_periods=24).mean()
top = U & (ret26.where(U).rank(axis=1, pct=True) > 0.9)
q = ups26.where(top).rank(axis=1, pct=True)
rows = []
for name, msk in (("top-decile 26wk winners, CHOPPY third (fewest up weeks)", q <= 1 / 3),
                  ("middle third", (q > 1 / 3) & (q <= 2 / 3)),
                  ("SMOOTH third (most up weeks)", q > 2 / 3)):
    for per, sl in (("dev", (C.index >= start) & (C.index < split)), ("holdout", C.index >= split)):
        row = {"group": name, "period": per}
        for h in (4, 13):
            exs = fwd[h].where(top & msk).sub(uni_fwd[h], axis=0)[sl]
            row[f"fwd{h}_excess_mean%"] = np.nanmean(exs.to_numpy()) * 100
            wk = exs.mean(axis=1).dropna()
            row[f"t{h}"] = wk.mean() / (wk.std() / np.sqrt(len(wk) / h))
        rows.append(row)
print("\nC. frog in the pan: among the top 10% by 26-week return, smooth vs choppy paths")
print(pd.DataFrame(rows).round(2).to_string(index=False))
