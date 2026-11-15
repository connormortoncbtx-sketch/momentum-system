"""
research/condor_skip_nn.py -- can a model reading the intraday chart up to entry spot the condor's losing days?
(Nov 15, 2026; Connor: point the CNN-style approach at the condor.)

The Oct 23 test (condor_skip.py) used only pre-10:00 information for the 10:00 condor and found NO usable skip rule:
"dangerous-looking" days were among the best, because the option market prices the danger. This test is about the
AFTERNOON slots we would actually run, and uses the day's own path up to entry.

PRE-REGISTERED (written before any result was seen):
  slots     noon put spread, 2pm condor, 3pm condor (k = 1 implied move, SPY $5 / QQQ 0.8% wings, held to close) --
            per-day return on risk from results/options_series_daily.parquet (real quotes; trade-bar proxy on gap days)
  inputs    only what is known at entry: today's 5-minute bars from 09:30 up to entry (intraday_liquid/5m_<und>),
            today's gap, yesterday's range, the 5- and 20-day mean daily range, VIX and VIX9D prior closes, weekday
  models    H   heuristic: skip when today's high-low range so far / the 20-day mean daily range is highest
            GBM gradient boosting on hand-made features (moves and ranges so far, all scaled by the 20-day mean range)
            NN  small 1D CNN over the 5-minute path (close/high/low vs the open, scaled by the 20-day mean range) plus
                the same scalar features; 3 seeds averaged
            all predict the trade's return on risk; walk-forward by calendar year: test year Y uses a model trained on
            2018..Y-1; test years 2020-2026
  skip      the riskiest 10% / 20% of days by prediction; the cut-off is the quantile of EARLIER out-of-sample predictions
            (the training-fit predictions for the first test year) -- no peeking at the test year
  book      per slot: 2% of NAV at risk on the SPY/QQQ average of the trades taken that day (skipped = flat)
  pass bar  a model counts only if it raises the book's Sharpe in BOTH 2020-23 and 2024-26, for BOTH the 2pm and the
            3pm condor, and beats H
Output: results/condor_skip_nn.csv (per slot x model x skip x period) and results/condor_skip_nn_preds.parquet.
"""
import os

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
SLOTS = {"noon_put": ("12:00", "put", 30), "2pm_condor": ("14:00", "condor", 54), "3pm_condor": ("15:00", "condor", 66)}
UNDS = ("SPY", "QQQ")
TEST_YEARS = range(2020, 2027)
SKIPS = (0.10, 0.20)
pd.set_option("display.width", 220)


def intraday(und):
    b = pd.read_parquet(f"{RD}/intraday_liquid/5m_{und}.parquet")
    b["ts"] = pd.to_datetime(b.ts)
    b["d"] = b.ts.dt.normalize()
    b["mins"] = b.ts.dt.hour * 60 + b.ts.dt.minute
    b = b[(b.mins >= 570) & (b.mins < 960)]
    day = b.groupby("d").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    day["rng"] = (day.h - day.l) / day.o
    day["prev_c"] = day.c.shift(1)
    day["rng20"] = day.rng.shift(1).rolling(20, min_periods=15).mean()
    day["rng5"] = day.rng.shift(1).rolling(5, min_periods=4).mean()
    day["rng1"] = day.rng.shift(1)
    return b, day


def vix():
    v = pd.read_parquet(f"{RD}/macro/cboe_VIX.parquet").assign(date=lambda x: pd.to_datetime(x.date)).set_index("date").value
    v9 = pd.read_parquet(f"{RD}/macro/cboe_VIX9D.parquet").assign(date=lambda x: pd.to_datetime(x.date)).set_index("date").value
    return pd.DataFrame({"vix": v, "vix9": v9}).sort_index().shift(1)        # prior close, aligned to the trade day


def build(und, entry, L, b, day, V):
    em = int(entry[:2]) * 60 + int(entry[3:])
    pre = b[b.mins < em]
    X_seq, rows = {}, []
    for d, g in pre.groupby("d"):
        if d not in day.index or not np.isfinite(day.at[d, "rng20"]) or len(g) < L * 0.8:
            continue
        o, s = day.at[d, "o"], day.at[d, "rng20"]
        seq = np.stack([(g.close / o - 1) / s, (g.high / o - 1) / s, (g.low / o - 1) / s]).astype("float32")[:, -L:]
        if seq.shape[1] < L:
            seq = np.pad(seq, ((0, 0), (L - seq.shape[1], 0)), mode="edge")
        X_seq[d] = seq
        r = g.close.pct_change().dropna()
        last = g.close.iloc[-1]
        hi, lo = g.high.max(), g.low.min()
        vv = V.reindex([d]).iloc[0] if d in V.index else pd.Series({"vix": np.nan, "vix9": np.nan})
        rows.append(dict(d=d, und=und,
                         range_so_far=(hi - lo) / o / s, move_so_far=(last / o - 1) / s, abs_move=abs(last / o - 1) / s,
                         gap=(o / day.at[d, "prev_c"] - 1) / s if np.isfinite(day.at[d, "prev_c"]) else 0.0,
                         rv_all=r.std() * np.sqrt(len(r)) / s, rv_last12=r.iloc[-12:].std() * np.sqrt(12) / s,
                         move_last6=(last / g.close.iloc[-7] - 1) / s if len(g) > 7 else 0.0,
                         pos_in_range=(last - lo) / (hi - lo) if hi > lo else 0.5,
                         rng1=day.at[d, "rng1"] / s, rng5=day.at[d, "rng5"] / s, rng20=s,
                         vix=vv.vix, vix_term=vv.vix9 / vv.vix if vv.vix else np.nan, dow=d.dayofweek))
    return pd.DataFrame(rows), X_seq


FEATS = ["range_so_far", "move_so_far", "abs_move", "gap", "rv_all", "rv_last12", "move_last6", "pos_in_range",
         "rng1", "rng5", "rng20", "vix", "vix_term", "dow", "is_qqq"]


def fit_gbm(tr, te):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                                      l2_regularization=1.0, random_state=0)
    m.fit(tr[FEATS], tr.y)
    return m.predict(tr[FEATS]), m.predict(te[FEATS])


def fit_nn(tr, te, seq, L):
    import torch
    import torch.nn as nn
    torch.set_num_threads(2)
    mu, sd = tr[FEATS].mean(), tr[FEATS].std().replace(0, 1)
    def tens(D):
        S = torch.tensor(np.stack([seq[(r.und, r.d)] for r in D.itertuples()]))
        F = torch.tensor(((D[FEATS] - mu) / sd).fillna(0).to_numpy("float32"))
        return S.clamp(-10, 10), F
    Str, Ftr = tens(tr); Ste, Fte = tens(te)
    ytr = torch.tensor(tr.y.to_numpy("float32"))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(3, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                                      nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
            self.head = nn.Sequential(nn.Linear(32 + len(FEATS), 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
        def forward(self, s, f):
            return self.head(torch.cat([self.conv(s).squeeze(-1), f], 1)).squeeze(-1)

    ptr, pte = [], []
    for seed in range(3):
        torch.manual_seed(seed)
        net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
        n = len(ytr)
        for ep in range(40):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, 128):
                j = perm[i:i + 128]
                loss = nn.functional.huber_loss(net(Str[j], Ftr[j]), ytr[j], delta=0.5)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            ptr.append(net(Str, Ftr).numpy()); pte.append(net(Ste, Fte).numpy())
    return np.mean(ptr, 0), np.mean(pte, 0)


def sharpe(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(252) if len(x) > 20 and x.std() > 0 else np.nan


def main():
    O = pd.read_parquet(f"{RD}/results/options_series_daily.parquet")
    O["d"] = pd.to_datetime(O.d)
    V = vix()
    intr = {u: intraday(u) for u in UNDS}
    out, allp = [], []
    for slot, (entry, struct, L) in SLOTS.items():
        sel = O[(O.entry == entry) & (O.struct == struct) & (O.k.astype(float) == 1.0) & (O.wing.astype(int) == 0) & (O.exit == "hold")]
        frames, seq = [], {}
        for u in UNDS:
            b, day = intr[u]
            F, S = build(u, entry, L, b, day, V)
            y = sel[sel.und == u].set_index("d").ror.astype(float)
            F = F[F.d.isin(y.index)].copy()
            F["y"] = F.d.map(y).clip(-1, 1)
            F["is_qqq"] = float(u == "QQQ")
            frames.append(F)
            seq.update({(u, d): v for d, v in S.items()})
        D = pd.concat(frames, ignore_index=True).sort_values("d").reset_index(drop=True)
        D = D[D.d >= "2018-01-01"]
        print(f"\n=== {slot}: {len(D):,} trades ({D.und.value_counts().to_dict()}), {D.d.min().date()}..{D.d.max().date()}; "
              f"mean RoR {D.y.mean() * 100:+.2f}%, max-loss (<= -90%) {(D.y <= -0.9).mean() * 100:.1f}%", flush=True)
        P = []
        for Y in TEST_YEARS:
            tr, te = D[D.d.dt.year < Y], D[D.d.dt.year == Y]
            if len(te) == 0 or len(tr) < 200:
                continue
            g_tr, g_te = fit_gbm(tr, te)
            n_tr, n_te = fit_nn(tr, te, seq, L)
            P.append(te.assign(H=-te.range_so_far, GBM=g_te, NN=n_te,
                               H_tr0=np.nan, GBM_tr0=np.quantile(g_tr, SKIPS[0]), NN_tr0=np.quantile(n_tr, SKIPS[0]),
                               GBM_tr1=np.quantile(g_tr, SKIPS[1]), NN_tr1=np.quantile(n_tr, SKIPS[1]),
                               H_trq0=np.quantile(-tr.range_so_far, SKIPS[0]), H_trq1=np.quantile(-tr.range_so_far, SKIPS[1])))
            print(f"  {Y}: train {len(tr):,} test {len(te):,} | rank corr pred vs outcome: "
                  + " ".join(f"{m} {pd.Series(P[-1][m]).corr(P[-1].y, method='spearman'):+.3f}" for m in ("H", "GBM", "NN")), flush=True)
        P = pd.concat(P, ignore_index=True).sort_values("d").reset_index(drop=True)
        P["slot"] = slot
        allp.append(P)
        for m in ("H", "GBM", "NN"):
            for si, q in enumerate(SKIPS):
                # cut-off: quantile of all EARLIER out-of-sample predictions (first year: training-fit predictions)
                cut = np.full(len(P), np.nan)
                first_cut = P[f"H_trq{si}"] if m == "H" else P[f"{m}_tr{si}"]
                vals = P[m].to_numpy()
                for i, d in enumerate(P.d):
                    prior = vals[(P.d < d).to_numpy() & (P.d.dt.year < d.year).to_numpy()]
                    cut[i] = np.quantile(prior, q) if len(prior) >= 150 else first_cut.iat[i]
                P[f"take_{m}_{si}"] = P[m] > cut
        for per, a, b in (("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-12-31")):
            W = P[(P.d >= a) & (P.d <= b)]
            base = W.groupby("d").y.mean() * 0.02
            out.append(dict(slot=slot, model="ALL (no skip)", skip=0, period=per, trades=len(W), sharpe=sharpe(base),
                            mean_ror=W.y.mean() * 100, pnl_kept=100.0, maxloss_avoided=0, maxloss_total=int((W.y <= -0.9).sum())))
            for m in ("H", "GBM", "NN"):
                for si, q in enumerate(SKIPS):
                    t = W[f"take_{m}_{si}"]
                    r = (W.y.where(t, 0.0)).groupby(W.d).mean() * 0.02
                    out.append(dict(slot=slot, model=m, skip=q, period=per, trades=int(t.sum()), sharpe=sharpe(r),
                                    mean_ror=W.y[t].mean() * 100, pnl_kept=W.y[t].sum() / W.y.sum() * 100,
                                    maxloss_avoided=int(((W.y <= -0.9) & ~t).sum()), maxloss_total=int((W.y <= -0.9).sum())))
    R = pd.DataFrame(out)
    R.to_csv(f"{RD}/results/condor_skip_nn.csv", index=False)
    pd.concat(allp).drop(columns=[c for c in pd.concat(allp).columns if c.endswith(("tr0", "tr1", "trq0", "trq1"))]) \
        .to_parquet(f"{RD}/results/condor_skip_nn_preds.parquet", index=False)
    piv = R.pivot_table(index=["slot", "model", "skip"], columns="period", values=["sharpe", "pnl_kept", "maxloss_avoided"])
    print("\n" + piv.round(2).to_string())
    print("\nmax-loss trades per slot/period:", R[R.model == "ALL (no skip)"].set_index(["slot", "period"]).maxloss_total.to_dict())


if __name__ == "__main__":
    main()
