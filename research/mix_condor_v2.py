"""
research/mix_condor_v2.py -- which momentum book pairs best with the condor overlay? (Oct 23, 2026)

Follow-up to mix_condor_momentum.py, whose momentum leg (LIVE proxy top 10, weekly, unhedged) was the weakest design
in the research (-4%/yr, -56% max DD over Feb 2024 - Oct 2026). Same condor leg (COMBO: SPY $5 / QQQ 0.8%, 0DTE,
1x implied move, risking f of equity per day on the spare margin, $0.01 and $0.03 per leg), now on top of:
  LIVE10      LIVE proxy top 10, weekly (the previous run)
  LIVE50      LIVE proxy top 50, weekly
  LIVE10_h    LIVE10 minus its dev-period (2017-23) IWM beta x IWM
  SLOW        12-1 momentum top 10, hold while top 100 (research/strategy_stack.py)
  SLOW_h      SLOW, IWM-hedged (research/hedged_stack.py)
  STACK_v2    shadow #7 v2: 0.104 SLOW_h + 0.238 CONTRACT_h + 0.504 ETF trend + 0.154 crypto trend, 8-week vol target
              (leverage 0.020698 / sd of the last 8 weekly returns, 0.5-3x, 6%/yr financing on the borrowed part)
Window: the condor data's span. Per book x cost x f: CAGR, vol, Sharpe, max DD, worst week, and correlation.
"""
import io, contextlib, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import alpha_v2 as A
from mix_condor_momentum import condor_daily, stats

pd.set_option("display.width", 240)


def live_books():
    p, U, R, live, ex1, ex4 = A.build()
    C = p.df("close")
    nxt = (C.shift(-1) / C - 1).clip(-0.9, 3)
    s = live.where(U)
    weeks = list(s.index)
    out = {10: {}, 50: {}}
    for i, w in enumerate(weeks[:-1]):
        if s.loc[w].notna().sum() < 150:
            continue
        for n in out:
            r = nxt.loc[w, list(s.loc[w].dropna().nlargest(n).index)].mean()
            if np.isfinite(r):
                out[n][weeks[i + 1]] = r - 2 * 0.0015
    return pd.Series(out[10], name="LIVE10"), pd.Series(out[50], name="LIVE50")


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        import hedged_stack as Hs
    l10, l50 = live_books()
    iwm = Hs.mk["IWM"]
    books = pd.concat([l10, l50], axis=1)
    books.index = pd.DatetimeIndex(books.index)
    iw = iwm.reindex(books.index)
    dev = (books.index < pd.Timestamp("2024-01-01")) & iw.notna() & books.LIVE10.notna()
    b = np.cov(books.LIVE10[dev], iw[dev])[0, 1] / iw[dev].var()
    books["LIVE10_h"] = books.LIVE10 - b * iw
    print(f"LIVE10 dev IWM beta {b:.2f}")
    books = books.join(Hs.R[["SLOW"]], how="outer").join(Hs.H[["SLOW_h"]], how="outer")
    st = (0.104 * Hs.H["SLOW_h"] + 0.238 * Hs.H["CONTRACT_h"] + 0.504 * Hs.R["ETFTREND"] + 0.154 * Hs.R["CRYPTO"].fillna(0)).dropna()
    lev = (0.020698 / st.rolling(8).std().shift(1)).clip(0.5, 3).fillna(1.0)
    books["STACK_v2"] = lev * st - (lev - 1).clip(lower=0) * 0.06 / 52

    res, corr = [], []
    for cost in (0.01, 0.03):
        cd = condor_daily(cost)
        fri = cd.index + pd.to_timedelta((4 - cd.index.dayofweek) % 7, unit="D")   # the Friday ending each day's week
        cw = {f: cd.groupby(fri).apply(lambda x: np.prod(1 + f * x.values) - 1) for f in (0.02, 0.05, 0.10)}
        win = books.index[(books.index >= cd.index.min()) & (books.index <= cd.index.max() + pd.Timedelta("6D"))]
        for name in ["LIVE10", "LIVE50", "LIVE10_h", "SLOW", "SLOW_h", "STACK_v2"]:
            m = books[name].reindex(win).dropna()
            if len(m) < 50:
                continue
            c2 = cw[0.02].reindex(m.index).fillna(0)
            corr.append(dict(book=name, cost=cost, weeks=len(m), corr_with_condors=np.corrcoef(m, c2)[0, 1],
                             condors_in_book_worst10pct=c2[m <= m.quantile(0.1)].mean() * 100))
            for f in (0, 0.02, 0.05, 0.10):
                c = cw[f].reindex(m.index).fillna(0) if f else 0
                res.append({**stats(m + c, name), "cost": cost, "condor_risk_per_day": f})
    R = pd.DataFrame(res)
    print("\nCORRELATIONS\n" + pd.DataFrame(corr).round(2).to_string(index=False))
    print("\nBOOK + CONDOR OVERLAY\n" + R.round(2).to_string(index=False))
    out = f"{A.RD}/results"
    os.makedirs(out, exist_ok=True)
    R.to_csv(f"{out}/mix_condor_v2.csv", index=False)
    pd.DataFrame(corr).to_csv(f"{out}/mix_condor_v2_corr.csv", index=False)
    for cost in (0.01, 0.03):
        parts = []
        for name in ["LIVE10", "LIVE50", "LIVE10_h", "SLOW", "SLOW_h", "STACK_v2"]:
            x = R[(R.cost == cost) & (R.book == name)].set_index("condor_risk_per_day")
            if len(x):
                parts.append(f"{name} {x.at[0, 'cagr_pct']:+.0f}%/{x.at[0, 'max_dd_pct']:.0f}% -> +2%: {x.at[0.02, 'cagr_pct']:+.0f}%/"
                             f"{x.at[0.02, 'max_dd_pct']:.0f}% Sh {x.at[0.02, 'sharpe']:.2f}")
        print(f"::notice::mix v2 cost {cost} (CAGR/maxDD alone -> with condors 2%/day): " + "; ".join(parts))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        print("::error::" + tb[-900:].replace("\n", " | "))
        raise
