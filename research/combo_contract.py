"""
research/combo_contract.py -- slow-momentum core + contract-win book, 2024-2026 (Oct 11, 2026).

Momentum core weekly returns (Mon close -> Mon close, research/combined_book.py sim_long)
and the contract book's daily returns (research_data/results/contract_book_daily.csv),
compounded over the same Monday-to-Monday windows. Blends w x contract + (1-w) x core,
then leverage at 1x .. full Kelly (financing 6%/yr on the borrowed part).
"early" = Jan 2024 - Mar 2025, "holdout" = Apr 2025 - Sep 2026.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("PANEL", "research_data")
import combined_book as CB          # runs its own tables first; we reuse sim_long and st

RD = os.environ.get("RESEARCH_DATA", os.environ["PANEL"])
d = pd.read_csv(f"{RD}/results/contract_book_daily.csv", index_col=0, parse_dates=True)["contract_book"]
core = CB.sim_long()
core = core[core.index >= "2023-12-29"]
rows = []
for fri in core.index:
    a, b = fri + pd.Timedelta(days=3), fri + pd.Timedelta(days=10)          # Mon close -> next Mon close
    x = d[(d.index > a) & (d.index <= b)]
    rows.append((fri, float((1 + x).prod() - 1) if len(x) else np.nan))
cw = pd.Series(dict(rows)).reindex(core.index)
df = pd.DataFrame({"core": core, "contract": cw}).dropna()
df = df[df.index >= "2024-01-01"]
SPLIT = pd.Timestamp("2025-04-01")
print("weekly correlation core vs contract:", round(df.core.corr(df.contract), 3))
FIN = 0.06 / 52


def st(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = len(r) / 52
    return dict(CAGR=(eq.iloc[-1] ** (1 / yrs) - 1) * 100 if eq.iloc[-1] > 0 else -100,
                Sharpe=r.mean() / r.std() * np.sqrt(52), maxDD=(eq / eq.cummax() - 1).min() * 100)


out = []
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    r = w * df.contract + (1 - w) * df.core
    e, h = r[r.index < SPLIT], r[r.index >= SPLIT]
    kst = (e - 0.04 / 52).mean() / (e - 0.04 / 52).var()          # Kelly from EARLY only
    for k in sorted({1.0, 1.5, 2.0, round(kst / 2, 2), round(kst, 2)}):
        lr = k * r - max(k - 1, 0) * FIN
        se, sh = st(lr[lr.index < SPLIT]), st(lr[lr.index >= SPLIT])
        out.append(dict(w_contract=w, leverage=k, kelly_early=round(kst, 2),
                        early_CAGR=se["CAGR"], early_Sharpe=se["Sharpe"], early_DD=se["maxDD"],
                        hold_CAGR=sh["CAGR"], hold_Sharpe=sh["Sharpe"], hold_DD=sh["maxDD"],
                        hold_weekly_geo=((1 + sh["CAGR"] / 100) ** (1 / 52) - 1) * 100))
pd.set_option("display.width", 250)
print(pd.DataFrame(out).round(2).to_string(index=False))
print("target: 3.00%/week geometric")
