"""
research/ira_stack.py -- (1) can the live 4-book stack (shadow #7) run inside an IRA, and
(2) does volatility targeting / drawdown de-levering help the taxable leveraged stack? (Oct 17, 2026)

Live mix (frozen in automation/shadow_stack.py): SLOW_h 14.5%, CONTRACT_h 35.3%, ETFTREND 29.4%,
CRYPTO 20.8%; hedges SLOW 0.798 x IWM, CONTRACT 0.743 x IWM.

IRA rules: no borrowing (no margin leverage), no short selling, no crypto.
  hedge   -> buy an inverse Russell 2000 ETF (RWM-style: -1x IWM DAILY, 0.95%/yr expense). Unlike a
             short, it costs cash: a book with weight w and beta b now needs w*(1+b) of capital,
             so all weights are rescaled to sum to 1 of CASH.
  crypto  -> spot BTC + ETH ETFs only (same 50-day trend rule on the two coins, 0.25%/yr fee).
  leverage-> none (1x).
Vol targeting (taxable, 2x average): weekly leverage = target / trailing realised vol (8, 13, 26 wks),
capped 0.5-3x, target set on dev so dev average leverage = 2. Drawdown rule: halve leverage
while the stack is > X% below its peak (X = 10, 15, 20). All chosen/compared on dev, holdout once.
"""
import contextlib, io, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    import hedged_stack as HS
S = HS.S
W = {"SLOW_h": 0.145, "CONTRACT_h": 0.353, "ETFTREND": 0.294, "CRYPTO": 0.208}
BETA = {"SLOW_h": 0.798, "CONTRACT_h": 0.743}
FIN = 0.06 / 52
SPLIT = S.SPLIT

R, mk = HS.R, HS.mk
taxable = pd.DataFrame({"SLOW_h": R.SLOW - BETA["SLOW_h"] * mk.IWM, "CONTRACT_h": R.CONTRACT - BETA["CONTRACT_h"] * mk.IWM,
                        "ETFTREND": R.ETFTREND, "CRYPTO": R.CRYPTO}).dropna()
stack1 = taxable @ pd.Series(W)

# --- IRA version ---------------------------------------------------------------------------
iwm_d = S.ret["IWM"].fillna(0)
inv = S.weekly_from_daily(-iwm_d - 0.0095 / 252).reindex(taxable.index)       # daily-reset -1x, expense
cp, cr = S.cp, S.cr
two = [c for c in cp.columns if c.split("/")[0] in ("BTC", "ETH")]
cv = cr[two].rolling(30, min_periods=20).std() * np.sqrt(365)
sig = (cp[two] > cp[two].rolling(50, min_periods=50).mean()).astype(float).shift(1)
wc = (sig * (0.40 / cv.shift(1)).clip(upper=1.5)).where(cp[two].notna())
wc = wc.div(wc.notna().sum(axis=1).clip(lower=1), axis=0).fillna(0)
rc2 = (wc * cr[two]).sum(axis=1) - wc.diff().abs().sum(axis=1) * 25 / 1e4 - wc.sum(axis=1) * 0.0025 / 365
cry2 = S.weekly_from_daily(rc2[rc2.index >= "2021-06-01"]).reindex(taxable.index).fillna(0)
# capital per unit of sleeve exposure: hedged books need 1 + beta of cash
cash = {"SLOW_h": 1 + BETA["SLOW_h"], "CONTRACT_h": 1 + BETA["CONTRACT_h"], "ETFTREND": 1.0, "CRYPTO": 1.0}
# ETF trend: weights sum to <= 4 gross across ETFs in the research book (vol-targeted). In an IRA gross
# must be <= 1 of cash, so scale it by the dev-average gross (measured below) -- conservative.
etf_gross = 1.0
ira_sleeves = pd.DataFrame({"SLOW_h": R.SLOW + BETA["SLOW_h"] * inv, "CONTRACT_h": R.CONTRACT + BETA["CONTRACT_h"] * inv,
                            "ETFTREND": R.ETFTREND, "CRYPTO": cry2}).reindex(taxable.index)
Wc = pd.Series({k: W[k] * cash[k] for k in W}); Wc = W and pd.Series(W) / Wc.sum()   # same exposure ratios, cash = 1
ira1 = ira_sleeves @ Wc

# ETF trend gross exposure check (the research book can exceed 100% of capital)
gross = S.w.abs().sum(axis=1)
print(f"ETF trend book gross exposure: mean {gross.mean():.2f}x, 95th pct {gross.quantile(0.95):.2f}x "
      f"(> 1x is not possible without margin -> IRA ETF sleeve scaled down)")
g = gross.resample("W-FRI").mean().reindex(taxable.index).ffill().clip(lower=1.0)
ira_sleeves_cap = ira_sleeves.copy(); ira_sleeves_cap["ETFTREND"] = R.ETFTREND.reindex(taxable.index) / g
ira1c = ira_sleeves_cap @ Wc

print("\nSTACK VERSIONS (weekly; dev 2017-23 | holdout 2024-26)")
for nm, r in (("taxable 1x (short IWM hedge, 16 coins)", stack1),
              ("taxable 2x (margin, 6% financing)", 2 * stack1 - FIN),
              ("IRA 1x: inverse-ETF hedge, BTC+ETH ETFs", ira1),
              ("IRA 1x, ETF sleeve capped at 1x gross", ira1c),
              ("IRA crypto sleeve alone (BTC+ETH)", cry2),
              ("taxable crypto sleeve alone (16 coins)", taxable.CRYPTO)):
    print(S.line(nm, r))
print("   IRA weights (cash): " + ", ".join(f"{k} {v:.3f}" for k, v in Wc.items()))

# --- vol targeting and drawdown rules on the taxable stack ----------------------------------
print("\nRISK OVERLAYS on the taxable stack (average leverage ~2x on dev; financing 6%/yr on borrowed part)")
base = stack1
dev = base.index < SPLIT
rows = [("fixed 2x", 2 * base - FIN, pd.Series(2.0, index=base.index))]
for lb in (8, 13, 26):
    rv = base.rolling(lb, min_periods=lb).std().shift(1)
    raw = 1 / rv
    k = 2.0 / raw[dev].mean()
    lev = (k * raw).clip(0.5, 3.0).fillna(2.0)
    k *= 2.0 / lev[dev].mean()                                  # re-hit 2x after the cap
    lev = (k * raw).clip(0.5, 3.0).fillna(2.0)
    rows.append((f"vol target, {lb}-wk lookback", lev * base - (lev - 1).clip(lower=0) * FIN, lev))
for X in (0.10, 0.15, 0.20):
    lev = []; eq = pk = 1.0
    for r in base:
        L = 1.0 if eq / pk - 1 < -X else 2.0
        lev.append(L); eq *= 1 + L * r - max(L - 1, 0) * FIN; pk = max(pk, eq)
    lev = pd.Series(lev, index=base.index)
    rows.append((f"2x, halve when DD > {X:.0%}", lev * base - (lev - 1).clip(lower=0) * FIN, lev))
for nm, r, lev in rows:
    print(S.line(nm, r) + f"  avg lev {lev[dev].mean():.2f}/{lev[~dev].mean():.2f}")
pd.DataFrame({nm: r for nm, r, _ in rows} | {"ira1": ira1, "ira1c": ira1c, "stack1": stack1}).to_csv(
    f"{S.P}/results/ira_stack_weekly.csv")
