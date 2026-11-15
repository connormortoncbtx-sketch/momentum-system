"""
research/intraday_momentum.py -- market intraday momentum (Gao, Han, Li & Zhou 2018) on SPY and QQQ, Nov 15, 2026.

PRE-REGISTERED: signal = return from the previous close to 10:00 ET (overnight + first half hour). Trade the last half
hour (15:30 -> 16:00 close) in the signal's direction. Costs 1.5 bp per side (3 bp round trip). Versions:
  long/short  sign(signal) x last-half-hour return
  long-only   last-half-hour return only on days with signal > 0 (cash account friendly)
Pass: long-only net Sharpe > 0.5 and t > 2 in BOTH 2017-23 and 2024-26. Also reported: predictive slope / t of the
last-half-hour return on the signal, and the 15:00-15:30 half hour as a second predictor (reported, not traded).
Data: intraday_liquid/5m_<und>.parquet (09:30-16:00 bars).
"""
import os

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
COST = 1.5e-4


def daily(und):
    b = pd.read_parquet(f"{RD}/intraday_liquid/5m_{und}.parquet")
    b["ts"] = pd.to_datetime(b.ts)
    b["d"] = b.ts.dt.normalize()
    b["m"] = b.ts.dt.hour * 60 + b.ts.dt.minute
    b = b[(b.m >= 570) & (b.m < 960)]
    g = b.groupby("d")
    out = pd.DataFrame({
        "close": g.close.last(),
        "p1000": b[b.m == 595].set_index("d").close,       # close of the 09:55 bar = 10:00 price
        "p1500": b[b.m == 895].set_index("d").close,       # 15:00
        "p1530": b[b.m == 925].set_index("d").close,       # 15:30
        "nbars": g.size()})
    out = out[out.nbars >= 70]                              # skip half days
    out["prev"] = out.close.shift(1)
    out["sig"] = out.p1000 / out.prev - 1
    out["h12"] = out.p1530 / out.p1500 - 1
    out["last"] = out.close / out.p1530 - 1
    return out.dropna()


def main():
    rows = []
    for und in ("SPY", "QQQ"):
        D = daily(und)
        D["ls"] = np.sign(D.sig) * D["last"] - 2 * COST
        D["lo"] = np.where(D.sig > 0, D["last"] - 2 * COST, 0.0)
        for lab, a, b in (("2017-23", "2017", "2023"), ("2024-26", "2024", "2026")):
            W = D.loc[a:b]
            beta = np.polyfit(W.sig, W["last"], 1)[0]
            res = W["last"] - beta * W.sig
            t_beta = beta / (res.std() / (W.sig.std() * np.sqrt(len(W))))
            b2 = np.polyfit(W.h12, W["last"], 1)[0]
            res2 = W["last"] - b2 * W.h12
            t2 = b2 / (res2.std() / (W.h12.std() * np.sqrt(len(W))))
            for v in ("ls", "lo"):
                x = W[v]
                traded = x[x != 0]
                rows.append(dict(und=und, period=lab, version={"ls": "long/short", "lo": "long-only"}[v], days=len(W),
                                 traded=len(traded), ann_ret_pct=x.mean() * 252 * 100, sharpe=x.mean() / x.std() * np.sqrt(252),
                                 t=x.mean() / x.std() * np.sqrt(len(x)), slope_sig=beta, t_slope_sig=t_beta, t_slope_h12=t2))
    R = pd.DataFrame(rows)
    R.to_csv(f"{RD}/results/intraday_momentum.csv", index=False)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
