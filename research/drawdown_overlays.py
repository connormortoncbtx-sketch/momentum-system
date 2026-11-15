"""
research/drawdown_overlays.py -- drawdown protection on the candidate books (Nov 15, 2026; Connor: "what can we do to
prevent those massive drawdowns?"). Volatility targeting, the one overlay that worked on Oct 17, applied to:
  FIPV  FIP-steady 12-1 momentum + CNN v2 veto (results/cnn_veto_series_v2.parquet, 2017+; veto active from 2022)
  COMBO equal-risk mix of FIPV, CONDOR, ETF_EQ (+ MERGER through 2023, CONTRACT from 2024) from combo_estimate.py
Rule: exposure = target vol / realised vol of the book's last L weekly returns (L = 8, 13), known before the week,
clipped 0.5-2.0x (Reg T); target = the book's own full-sample vol, so average exposure is ~1x (a pure timing test).
Reported per period: CAGR, Sharpe, max drawdown, worst 13-week return. Public data only.
"""
import os

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
SPLIT = pd.Timestamp("2024-01-01")


def st(r):
    r = r.dropna()
    eq = (1 + r).cumprod(); y = len(r) / 52
    return dict(CAGR=(eq.iloc[-1] ** (1 / y) - 1) * 100, Sharpe=r.mean() / r.std() * np.sqrt(52),
                maxDD=(eq / eq.cummax() - 1).min() * 100, worst13w=((1 + r).rolling(13).apply(np.prod, raw=True) - 1).min() * 100)


def vt(r, L, target):
    lev = (target / r.rolling(L).std().shift(1)).clip(0.5, 2.0)
    return (r * lev).dropna(), lev


def main():
    S = {}
    S["FIPV"] = pd.read_parquet(f"{RD}/results/cnn_veto_series_v2.parquet")["FIP_M12_1|10"].dropna()
    C = pd.read_csv(f"{RD}/results/combo_streams_weekly.csv", index_col=0, parse_dates=True)
    cols = [c for c in ("FIPV", "CONDOR", "ETF_EQ", "CONTRACT") if c in C]
    Z = C[cols].loc["2018-01-01":]
    z = Z / Z.std()                                    # equal risk per stream (full-sample vol; a mild look-ahead, shown as such)
    S["COMBO (equal risk, avail. streams)"] = (z.mean(axis=1) * Z.std().mean()).dropna()
    rows = []
    for name, r in S.items():
        tgt = r.std()
        for lab, rr, lev in [("none", r, pd.Series(1.0, index=r.index))] + [(f"vol target {L}w", *vt(r, L, tgt)) for L in (8, 13)]:
            for per, sel in (("2017/18-2023", rr.index < SPLIT), ("2024-26", rr.index >= SPLIT)):
                rows.append(dict(book=name, overlay=lab, period=per, avg_exposure=float(lev.reindex(rr.index)[sel].mean()), **st(rr[sel])))
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(R.round(2).to_string(index=False))
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/drawdown_overlays.csv", index=False)


if __name__ == "__main__":
    main()
