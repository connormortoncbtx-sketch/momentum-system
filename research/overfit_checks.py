"""
research/overfit_checks.py -- how much of our backtest performance is selection luck? (Nov 15, 2026; built ahead of the
Jan 5 review so that day is mechanical: rerun this, read the table.)

Two standard tools (Bailey & Lopez de Prado):
  DEFLATED SHARPE RATIO (DSR, 2014): the probability that the true Sharpe is above zero AFTER accounting for how many
      variants were tried (N), how much their Sharpes varied, the track-record length, and fat tails / skew. The
      benchmark SR0 is the Sharpe the BEST of N worthless variants would show by luck:
          SR0 = sqrt(V) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e))),  g = Euler-Mascheroni 0.5772
          DSR = Phi( (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2) )        (per-period SR)
      DSR >= 0.95 is the usual bar.
  PROBABILITY OF BACKTEST OVERFITTING (PBO, CSCV 2015): split the history into S blocks; for every way of using half the
      blocks to pick the best variant, check where that pick ranks in the other half. PBO = share of splits where the
      in-sample winner lands BELOW the out-of-sample median. < 0.2 is reassuring, ~0.5 means picking is no better than
      chance. Needs every variant's return series -- we have that for the option-premium family (360 variants, daily).
Usage: python research/overfit_checks.py   (RESEARCH_DATA set). Writes results/overfit_checks.csv.
"""
import itertools
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

RD = os.environ.get("RESEARCH_DATA", "research_data")
EG = 0.5772156649
pd.set_option("display.width", 220)


def sr0(n_trials, var_sr):
    """Expected max per-period Sharpe of n_trials zero-skill variants whose Sharpes have variance var_sr."""
    if n_trials <= 1:
        return 0.0
    return math.sqrt(var_sr) * ((1 - EG) * norm.ppf(1 - 1 / n_trials) + EG * norm.ppf(1 - 1 / (n_trials * math.e)))


def dsr(r, n_trials, var_sr):
    """Deflated Sharpe ratio of per-period returns r. var_sr is the variance of PER-PERIOD Sharpes across the trials."""
    r = pd.Series(r).dropna()
    T = len(r)
    s = r.mean() / r.std()
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    b = sr0(n_trials, var_sr)
    den = math.sqrt(max(1 - g3 * s + (g4 - 1) / 4 * s * s, 1e-9))
    return float(norm.cdf((s - b) * math.sqrt(T - 1) / den)), s, b, T, g3, g4


def pbo(M, S=16):
    """CSCV probability of backtest overfitting for a T x N matrix of per-period returns (columns = variants).
    Returns PBO, the median OOS Sharpe percentile of the IS winner, and mean IS vs OOS Sharpe of the winner."""
    X = np.asarray(M, dtype="float64")
    T, N = X.shape
    edges = np.linspace(0, T, S + 1).astype(int)
    n = np.array([edges[i + 1] - edges[i] for i in range(S)], dtype="float64")
    s1 = np.array([X[edges[i]:edges[i + 1]].sum(0) for i in range(S)])
    s2 = np.array([(X[edges[i]:edges[i + 1]] ** 2).sum(0) for i in range(S)])

    def sharpe(idx):
        nn, a, b = n[idx].sum(), s1[idx].sum(0), s2[idx].sum(0)
        mu = a / nn
        sd = np.sqrt(np.maximum(b / nn - mu * mu, 1e-18))
        return mu / sd

    logits, pct, is_sr, oos_sr = [], [], [], []
    for comb in itertools.combinations(range(S), S // 2):
        ins = list(comb)
        oos = [i for i in range(S) if i not in comb]
        a, b = sharpe(ins), sharpe(oos)
        k = int(np.argmax(a))
        w = (b < b[k]).sum() / (N - 1)                       # OOS percentile of the IS winner (0..1)
        w = min(max(w, 1 / N), 1 - 1 / N)
        logits.append(math.log(w / (1 - w)))
        pct.append(w); is_sr.append(a[k]); oos_sr.append(b[k])
    L = np.array(logits)
    return float((L <= 0).mean()), float(np.median(pct)), float(np.mean(is_sr)), float(np.mean(oos_sr))


def option_family():
    o = pd.read_parquet(f"{RD}/results/options_series_daily.parquet")
    o["cfg"] = (o.entry.astype(str) + "|" + o.struct.astype(str) + "|k" + o.k.astype(float).astype(str) + "|w"
                + o.wing.astype(int).astype(str) + "|" + o.exit.astype(str))
    M = o.pivot_table(index="d", columns="cfg", values="ror", aggfunc="mean").sort_index()   # SPY+QQQ average
    M = M.loc[:, M.notna().mean() > 0.9].fillna(0.0) * 0.02                                  # 2% of NAV at risk a day
    return M


def main():
    rows = []
    # ---- option-premium family: PBO over all 360 saved variants, DSR for the picks
    M = option_family()
    print(f"OPTION FAMILY: {M.shape[1]} variants x {M.shape[0]} days {M.index.min().date()}..{M.index.max().date()}")
    sr_all = M.mean() / M.std()
    var_sr = float(sr_all.var())
    print(f"  per-day Sharpe across variants: median {sr_all.median() * math.sqrt(252):.2f}/yr, "
          f"sd {math.sqrt(var_sr) * math.sqrt(252):.2f}/yr")
    for lab, a, b in (("2018-26", None, None), ("2018-23", None, "2023-12-31"), ("2024-26", "2024-01-01", None)):
        P = pbo(M.loc[a:b])
        print(f"  PBO {lab}: {P[0]:.2f} | IS winner's median OOS percentile {P[1]:.2f} | winner Sharpe IS {P[2] * math.sqrt(252):.2f} -> OOS {P[3] * math.sqrt(252):.2f}/yr")
        rows.append(dict(test="PBO", family="options (360 variants)", window=lab, value=round(P[0], 3),
                         note=f"IS-best OOS pct {P[1]:.2f}; Sharpe IS {P[2] * math.sqrt(252):.2f} -> OOS {P[3] * math.sqrt(252):.2f}"))
    picks = {"3pm condor k1": "15:00|condor|k1.0|w0|hold", "2pm condor k1": "14:00|condor|k1.0|w0|hold",
             "noon put k1": "12:00|put|k1.0|w0|hold"}
    picks = {k: v for k, v in picks.items() if v in M}
    trio = M[list(picks.values())].mean(axis=1) if len(picks) == 3 else None
    series = {k: M[v] for k, v in picks.items()}
    if trio is not None:
        series["trio (equal split)"] = trio
    for name, r in series.items():
        for N in (360, 7560):                                 # saved variants / every cell of the full options grid
            for lab, a, b in (("2018-26", None, None), ("2024-26", "2024-01-01", None)):
                d, s, b0, T, g3, g4 = dsr(r.loc[a:b], N, var_sr)
                rows.append(dict(test="DSR", family="options", window=lab, value=round(d, 3),
                                 note=f"{name}; N={N}; SR {s * math.sqrt(252):.2f}/yr vs luck bar {b0 * math.sqrt(252):.2f}/yr; T={T}; skew {g3:.1f}; kurt {g4:.0f}"))
    # ---- stock book (FIP 12-1 + CNN veto) and the combination: DSR at several honest trial counts
    C = pd.read_csv(f"{RD}/results/combo_streams_weekly.csv", index_col=0, parse_dates=True)
    G = pd.read_csv(f"{RD}/results/growth_yield_grid.csv")                 # 256 stock-book variants: Sharpe dispersion
    var_w = float((G.dev_Sharpe / math.sqrt(52)).var())
    stock = {"FIPV (FIP 12-1 + CNN veto)": C.FIPV.dropna(),
             "FIPV + CONDOR equal risk": ((C[["FIPV", "CONDOR"]] / C[["FIPV", "CONDOR"]].std()).mean(axis=1)
                                          .where(C[["FIPV", "CONDOR"]].notna().all(axis=1)).dropna())}
    print(f"\nSTOCK BOOKS: trial Sharpe sd (from the 256-variant grid) {math.sqrt(var_w) * math.sqrt(52):.2f}/yr")
    # The grid's dispersion is narrow (one family of similar rules); the stock variants actually tried in Oct-Nov ranged
    # from Sharpe ~0 to ~1.2, so a 0.4/yr spread is the realistic case. Report both.
    for disp, vv in (("grid sd", var_w), ("sd 0.40", (0.40 / math.sqrt(52)) ** 2)):
        for name, r in stock.items():
            for N in (50, 200, 1000):
                for lab, a, b in (("2018-23", "2018-01-01", "2023-12-31"), ("2024-26", "2024-01-01", None), ("2018-26", "2018-01-01", None)):
                    d, s, b0, T, g3, g4 = dsr(r.loc[a:b], N, vv)
                    rows.append(dict(test="DSR", family=f"stocks ({disp})", window=lab, value=round(d, 3),
                                     note=f"{name}; N={N}; SR {s * math.sqrt(52):.2f}/yr vs luck bar {b0 * math.sqrt(52):.2f}/yr; T={T}; skew {g3:.1f}; kurt {g4:.0f}"))
    R = pd.DataFrame(rows)
    R.to_csv(f"{RD}/results/overfit_checks.csv", index=False)
    print("\n" + R.to_string(index=False))


if __name__ == "__main__":
    main()
