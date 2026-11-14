"""
research/leadlag.py -- clustering and lead-lag: do laggards catch up when the stocks that move WITH them rally?
(Nov 13, 2026; Connor approved the "more sophisticated methods" program, step 1.)

Evidence in the literature: industry/peer lead-lag (Hou 2007), economic links (Cohen & Frazzini 2008), "connected"
stocks by return co-movement. Here peers come from the data, not from sector labels:
  every 13 weeks, using only the trailing 104 weeks (>= 78 observed) of MARKET-RESIDUAL weekly returns:
    kNN10 / kNN25   each stock's 10 / 25 most correlated stocks
    KM100           k-means into 100 clusters on the standardized residual return paths (= correlation clustering)
Signals at week t (known at Friday's close; entry Monday close, the usual convention):
    PEER1  peers' average return last week                 (pure "the group just moved")
    PEER4  peers' average return over the last 4 weeks
    GAP1   PEER1 minus the stock's own last-week return       (laggard of a moving group)
    GAP4   PEER4 minus own 4-week return
    OWNREV minus own last-week return                         (control: plain short-term reversal)
Scoring: weekly rank IC with the next week's return; top-decile minus universe (next 1 week and next 4 weeks);
top-10 / top-30 weekly books with spread costs. Dev 2018-2023 (clusters need 2 years) / holdout 2024-Sep 2026 (the
holdout lacks most delistings -- being fixed). Universe price >= $5, ADV >= $2M. Public data only.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as B  # noqa: E402
import universe_floor as U  # noqa: E402

RD = os.environ.get("RESEARCH_DATA", "research_data")
SPLIT = pd.Timestamp("2024-01-01")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)


def main():
    p = B.load_panel(RD); f = B.features(p)
    base = (f["hist"] >= 53) & ~pd.DataFrame(np.tile(p.is_fund, (len(p.weeks), 1)), index=p.weeks, columns=p.symbols) \
        & ~f["rsplit_52w"].fillna(False).astype(bool)
    m = base & (f["price"] >= 5) & (f["adv"] >= 2e6)
    C = p.df("close")
    r1 = (C / C.shift(1) - 1).where(lambda x: x.abs() < 1.0)
    r4 = (C / C.shift(4) - 1).where(lambda x: x.abs() < 2.0)
    mkt = r1.where((f["price"] >= 5) & (f["adv"] >= 2e6)).mean(axis=1)   # market from day one (m needs 53 weeks)
    res = r1.sub(mkt, axis=0)
    d1c = p.df("d1_close")
    f1 = (d1c.shift(-2) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 1.5)
    f4 = (d1c.shift(-5) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 3)
    weeks, syms = p.weeks, np.array(p.symbols)
    sig = {k: pd.DataFrame(np.nan, index=weeks, columns=syms) for k in
           ("kNN10_PEER1", "kNN25_PEER1", "KM100_PEER1", "kNN10_PEER4", "KM100_PEER4",
            "kNN10_GAP1", "kNN25_GAP1", "KM100_GAP1", "kNN10_GAP4", "KM100_GAP4")}
    R1 = r1.to_numpy(); R4 = r4.to_numpy(); RES = res.to_numpy(); M = m.to_numpy()
    starts = [i for i in range(104, len(weeks)) if (i - 104) % 13 == 0]
    for si, i in enumerate(starts):
        j_end = starts[si + 1] if si + 1 < len(starts) else len(weeks)
        win = RES[i - 104:i]                                     # weeks before the formation date only
        ok = (np.isfinite(win).sum(axis=0) >= 78) & M[i - 1]
        idx = np.where(ok)[0]
        X = np.nan_to_num(win[:, idx]).T                          # stocks x weeks
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
        corr = X @ X.T / X.shape[1]
        np.fill_diagonal(corr, -np.inf)
        order = np.argsort(-corr, axis=1)
        km = KMeans(n_clusters=100, n_init=3, random_state=0).fit(X).labels_
        for j in range(i, j_end):                                 # apply until the next re-formation
            a1, a4 = R1[j, idx], R4[j, idx]
            f1n, f4n = np.isfinite(a1), np.isfinite(a4)
            z1, z4 = np.where(f1n, a1, 0.0), np.where(f4n, a4, 0.0)
            out = {}
            for k in (10, 25):
                nb = order[:, :k]
                p1 = z1[nb].sum(1) / np.maximum(f1n[nb].sum(1), 1)
                p4 = z4[nb].sum(1) / np.maximum(f4n[nb].sum(1), 1)
                out[f"kNN{k}_PEER1"], out[f"kNN{k}_GAP1"] = p1, p1 - a1
                if k == 10:
                    out["kNN10_PEER4"], out["kNN10_GAP4"] = p4, p4 - a4
            s1 = np.bincount(km, z1, 100); n1 = np.bincount(km, f1n.astype(float), 100)
            s4 = np.bincount(km, z4, 100); n4 = np.bincount(km, f4n.astype(float), 100)
            c1 = (s1[km] - z1) / np.maximum(n1[km] - f1n, 1)       # cluster mean excluding the stock itself
            c4 = (s4[km] - z4) / np.maximum(n4[km] - f4n, 1)
            out["KM100_PEER1"], out["KM100_GAP1"] = c1, c1 - a1
            out["KM100_PEER4"], out["KM100_GAP4"] = c4, c4 - a4
            for k, v in out.items():
                sig[k].iloc[j, idx] = v
        print(f"  formation {weeks[i].date()}: {len(idx)} stocks", flush=True)
    sig["OWNREV1"] = -r1
    sig["OWNREV4"] = -r4
    small = (f["price"] < 5) | (f["adv"] < 2e6)
    cost = pd.DataFrame(U.half_spread_bps(f["price"].fillna(0).to_numpy(), f["adv"].fillna(0).to_numpy()), index=weeks, columns=syms)
    first = weeks[starts[0]]
    rows = []
    for name, s in sig.items():
        s = s.where(m).loc[first:]
        ic = s.rank(axis=1).corrwith(f1.loc[first:].where(m.loc[first:]).rank(axis=1), axis=1)
        top = s.rank(axis=1, pct=True) >= 0.9
        ex1 = f1.loc[first:].where(top).mean(axis=1) - f1.loc[first:].where(m.loc[first:]).mean(axis=1)
        ex4 = f4.loc[first:].where(top).mean(axis=1) - f4.loc[first:].where(m.loc[first:]).mean(axis=1)
        row = dict(signal=name)
        for lab, sel in (("dev", ic.index < SPLIT), ("hold", ic.index >= SPLIT)):
            x = ic[sel].dropna()
            row[f"{lab}_IC"] = x.mean(); row[f"{lab}_ICt"] = x.mean() / x.std() * np.sqrt(len(x))
            row[f"{lab}_top10pct_1w_bp"] = ex1[sel].mean() * 1e4
            row[f"{lab}_top10pct_4w_bp"] = ex4[sel].mean() * 1e4
        for n in (10, 30):
            r = U.sim(s, m.loc[first:], f1.loc[first:], cost.loc[first:], small.loc[first:], n=n, k=1)
            d, h = U.st(r[r.index < SPLIT]), U.st(r[r.index >= SPLIT])
            row[f"top{n}_dev_CAGR"], row[f"top{n}_dev_Sh"] = d["CAGR"], d["Sharpe"]
            row[f"top{n}_hold_CAGR"], row[f"top{n}_hold_Sh"] = h["CAGR"], h["Sharpe"]
        rows.append(row)
        print(f"  scored {name}", flush=True)
    R = pd.DataFrame(rows)
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/leadlag.csv", index=False)
    print("\nRANK IC with next week's return (t-stat over weeks); top-decile excess vs universe in basis points:")
    print(R[["signal", "dev_IC", "dev_ICt", "hold_IC", "hold_ICt", "dev_top10pct_1w_bp", "hold_top10pct_1w_bp",
             "dev_top10pct_4w_bp", "hold_top10pct_4w_bp"]].round(4).to_string(index=False))
    print("\nWEEKLY BOOKS after spread costs (CAGR %, Sharpe):")
    print(R[["signal"] + [c for c in R.columns if c.startswith("top")]].round(2).to_string(index=False))
    b = R.sort_values("dev_ICt", ascending=False).iloc[0]
    print(f"::notice::lead-lag: best dev IC {b.signal} t {b.dev_ICt:.1f} -> holdout t {b.hold_ICt:.1f}")


if __name__ == "__main__":
    main()
