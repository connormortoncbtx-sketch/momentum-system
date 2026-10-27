"""
research/crypto4.py -- the crypto sleeve through spot ETFs (Oct 27, 2026).

Connor's live Alpaca account (Texas) can't trade crypto; only coins with a US spot ETF can be held: BTC (IBIT), ETH (ETHA),
SOL (BSOL), XRP. Three versions of momentum v2's crypto sleeve (same rule: coin above its 50-day average, 40% vol
target, gross <= 1, 25 bps per unit traded, idle cash at T-bills), each dropped into the live mix (W2 weights):
  C16        the sleeve as backtested and shadowed: all 16 coins
  C4 faithful  same per-coin weights as C16 (divided by the 16-coin count), but only BTC/ETH/SOL/XRP held -> the
             other slots are cash. This is what live trades from Oct 27.
  C4 renorm  the rule run on the 4 coins alone (divided by 4) -- more of the sleeve invested
ETF versions pay a 0.25%/yr fund fee on what is held. Dev < 2024 / holdout 2024-26, as stack_v2.
"""
import contextlib, io, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    import stack_v2 as V
S = V.S
FOUR = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]
W2 = {"SLOW_h": 0.104, "CONTRACT_h": 0.238, "ETFTREND": 0.504, "CRYPTO": 0.154}
pd.set_option("display.width", 200)


def sleeve(cols, divisor_cols, fee=0.0):
    cp, cr = S.cp, S.cr
    cv = cr.rolling(30, min_periods=20).std() * np.sqrt(365)
    sig = (cp > cp.rolling(50, min_periods=50).mean()).astype(float).shift(1)
    wc = (sig * (0.40 / cv.shift(1)).clip(upper=1.5)).where(cp.notna())
    n = wc[divisor_cols].notna().sum(axis=1).clip(lower=1)
    wc = wc[cols].div(n, axis=0).fillna(0)
    gc = wc.sum(axis=1); wc = wc.div(gc.clip(lower=1), axis=0); gc = wc.sum(axis=1)
    d = (wc * cr[cols].fillna(0)).sum(axis=1) - wc.diff().abs().sum(axis=1) * 25 / 1e4 \
        + (1 - gc) * V.rf_daily(wc.index) * 252 / 365 - gc * fee / 365
    return S.weekly_from_daily(d[d.index >= "2021-06-01"]), gc


def main():
    allc = list(S.cp.columns)
    four = [c for c in FOUR if c in allc]
    print(f"coins in data: {len(allc)}; ETF-able: {four}")
    variants = {"C16 (backtest/shadow)": sleeve(allc, allc),
                "C4 faithful (live now)": sleeve(four, allc, 0.0025),
                "C4 renorm": sleeve(four, four, 0.0025)}
    A = V.A.copy()
    rows = []
    for nm, (cry, gc) in variants.items():
        c = cry.reindex(A.index).fillna(V.rf_weekly(A.index))
        print(V.line(f"sleeve {nm}", c) + f"  avg invested {gc[gc.index >= '2021-06-01'].mean():.0%}")
        A2 = A.assign(CRYPTO=c)
        book = sum(A2[k] * w for k, w in W2.items())
        print(V.line(f"BOOK  {nm}", book))
        for part, r in (("dev", book[book.index < V.SPLIT]), ("hold", book[book.index >= V.SPLIT]),
                        ("crypto era", book[book.index >= "2021-06-01"])):
            ex = r - V.RF.reindex(r.index)
            rows.append(dict(variant=nm, part=part, ann_ret=r.mean() * 52 * 100, ex_sharpe=ex.mean() / ex.std() * np.sqrt(52)))
    R = pd.DataFrame(rows).pivot(index="variant", columns="part", values=["ann_ret", "ex_sharpe"]).round(2)
    print("\n" + R.to_string())
    os.makedirs(f"{S.P}/results", exist_ok=True)
    R.to_csv(f"{S.P}/results/crypto4.csv")
    g = lambda v, p: R.loc[v, ("ex_sharpe", p)]
    print(f"::notice::crypto via ETFs, book excess Sharpe dev/hold: C16 {g('C16 (backtest/shadow)', 'dev'):.2f}/{g('C16 (backtest/shadow)', 'hold'):.2f}, "
          f"C4 faithful {g('C4 faithful (live now)', 'dev'):.2f}/{g('C4 faithful (live now)', 'hold'):.2f}, "
          f"C4 renorm {g('C4 renorm', 'dev'):.2f}/{g('C4 renorm', 'hold'):.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
