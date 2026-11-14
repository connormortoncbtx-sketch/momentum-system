"""
research/chart_cnn2.py -- chart-image CNN, version 2 (Nov 14, 2026; approved "Do it").

Changes from chart_cnn.py (v1), all aimed at the use we found -- a veto on collapses:
  1. More, fresher training data: samples from EVERY trading day of 2017-2021 (v1: Fridays only, 30% kept), randomly
     thinned to MAX_TRAIN (default 600k) to fit a free 4-core runner.
  2. Two targets, one shared network: (a) beat the median stock over the next 5 trading days (v1's target);
     (b) COLLAPSE: land in the bottom 10% of all stocks over the next 20 trading days. The veto uses (b).
  3. One seed per run (SEED env) so two runs can train in parallel within the 6-hour job limit; ensemble afterwards.
Same images (research/chart_images.py), same universe (price >= $5, 20-day $ volume >= $2M, funds excluded,
delisted stocks included), same split: train 2017-21, early stopping on 2022-23, TEST 2024+ never used in training.
Predictions for EVERY Friday 2022+ (both periods out of sample) -> results/chart_cnn2_preds_seed<S>.parquet;
weights -> results/chart_cnn2_seed<S>.pt. Env: SEED (0), MAX_TRAIN (600000), EPOCHS (3), LIMIT (smoke test).
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chart_images as CI  # noqa: E402

RD = os.environ.get("RESEARCH_DATA", "research_data")
SEED = int(os.environ.get("SEED") or 0)
MAX_TRAIN = int(os.environ.get("MAX_TRAIN") or 600000)
EPOCHS = int(os.environ.get("EPOCHS") or 3)
LIMIT = int(os.environ.get("LIMIT") or 0)
TRAIN_START, TRAIN_END, VAL_END = pd.Timestamp("2017-01-01"), pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")


def load():
    D = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{RD}/daily/*ohlcv_*.parquet"))], ignore_index=True)
    D = D.dropna(subset=["open", "high", "low", "close"])
    D = D[(D.close > 0) & (D.high >= D.low)]
    try:
        import backtest as B
        p = B.load_panel(RD)
        D = D[~D.symbol.isin({s for s, isf in zip(p.symbols, p.is_fund) if isf})]
    except Exception as e:
        print("fund filter skipped:", e)
    if LIMIT:
        D = D[D.symbol.isin(sorted(D.symbol.unique())[:LIMIT])]
    return D.sort_values(["symbol", "date"])


def build(D):
    days = np.array(sorted(D.date.unique()))
    di = pd.DatetimeIndex(days)
    is_fri = np.zeros(len(days), bool)
    is_fri[pd.Series(np.arange(len(days)), index=di).groupby(di.to_period("W-FRI")).max().values] = True
    n_train_days = ((di >= TRAIN_START) & (di < TRAIN_END)).sum()
    expect = n_train_days * 2600
    p_train = min(1.0, MAX_TRAIN / max(expect, 1))
    rng = np.random.default_rng(100 + SEED)
    imgs, meta = [], []
    t0 = time.time()
    for n, (sym, g) in enumerate(D.groupby("symbol", sort=False)):
        if len(g) < 60:
            continue
        dt_ = pd.DatetimeIndex(g.date.to_numpy())
        pos = days.searchsorted(dt_.values)
        o, hi, lo, c, v = (g[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "volume"))
        ma = pd.Series(c).rolling(20).mean().to_numpy()
        dv = pd.Series(c * v).rolling(20).mean().to_numpy()
        ks = np.arange(39, len(c) - 21)
        ks = ks[(c[ks] >= 5) & (dv[ks] >= 2e6)]
        dd = dt_.values[ks]
        fr = is_fri[pos[ks]]
        in_tr = (dd >= TRAIN_START.to_datetime64()) & (dd < TRAIN_END.to_datetime64())
        keep = (in_tr & (rng.random(len(ks)) <= p_train)) | (~in_tr & (dd >= TRAIN_END.to_datetime64()) & fr)
        for k, d in zip(ks[keep], dd[keep]):
            part = 0 if d < TRAIN_END.to_datetime64() else (1 if d < VAL_END.to_datetime64() else 2)
            s = slice(k - 19, k + 1)
            img = CI.draw(o[s], hi[s], lo[s], c[s], ma[s], v[s])
            if img is None:
                continue
            imgs.append(img)
            meta.append((sym, pd.Timestamp(d), part, c[k + 6] / c[k + 1] - 1, c[k + 21] / c[k + 1] - 1, c[k], dv[k]))
        if n % 1000 == 0:
            print(f"  images: {n} symbols, {len(imgs):,} samples, {time.time() - t0:.0f}s", flush=True)
    M = pd.DataFrame(meta, columns=["symbol", "date", "part", "fwd5", "fwd20", "price", "dv20"])
    for col in ("fwd5", "fwd20"):
        M[col] = M[col].where(M[col].abs() < 3)
    M["y_up"] = (M.fwd5 > M.groupby("date").fwd5.transform("median")).astype(np.int64)
    M["y_crash"] = (M.groupby("date").fwd20.rank(pct=True) <= 0.10).astype(np.int64)
    return np.stack(imgs), M


def net2():
    import torch.nn as nn
    trunk = CI.net()
    trunk[-1] = nn.Linear(64 * 8 * 60, 4)          # logits: [up: no, yes, crash: no, yes]
    return trunk


def train(X, M, Xv, Mv):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.set_num_threads(os.cpu_count() or 4)
    model = net2(); opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    ce = nn.CrossEntropyLoss()
    yu, yc = M.y_up.to_numpy(), M.y_crash.to_numpy()
    yuv, ycv = Mv.y_up.to_numpy(), Mv.y_crash.to_numpy()
    tv = lambda a: torch.from_numpy(a[:, None].astype(np.float32) / 255.0)
    w_crash = torch.tensor([1.0, 4.0])              # the crash class is 10% of samples
    ce_c = nn.CrossEntropyLoss(weight=w_crash)
    best, state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train(); perm = np.random.permutation(len(X)); t0 = time.time(); tot = 0.0
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]
            out = model(tv(X[b]))
            loss = ce(out[:, :2], torch.from_numpy(yu[b])) + ce_c(out[:, 2:], torch.from_numpy(yc[b]))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for i in range(0, len(Xv), 2048):
                out = model(tv(Xv[i:i + 2048]))
                vl += float(ce(out[:, :2], torch.from_numpy(yuv[i:i + 2048])) +
                            ce_c(out[:, 2:], torch.from_numpy(ycv[i:i + 2048]))) * len(Xv[i:i + 2048])
        vl /= len(Xv)
        print(f"  seed {SEED} epoch {ep + 1}: train {tot / len(X):.4f}, val {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 1:
                break
    model.load_state_dict(state); model.eval()
    return model


def predict(model, X):
    import torch
    pu, pc = np.zeros(len(X)), np.zeros(len(X))
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            out = model(torch.from_numpy(X[i:i + 4096][:, None].astype(np.float32) / 255.0))
            pu[i:i + 4096] = torch.softmax(out[:, :2], 1)[:, 1].numpy()
            pc[i:i + 4096] = torch.softmax(out[:, 2:], 1)[:, 1].numpy()
    return pu, pc


def main():
    t0 = time.time()
    D = load()
    print(f"daily rows {len(D):,}, symbols {D.symbol.nunique():,}", flush=True)
    X, M = build(D); del D
    ok = (M.fwd5.notna() & M.fwd20.notna()).to_numpy()
    X, M = X[ok], M[ok].reset_index(drop=True)
    tr, va, te = (M.part == 0).to_numpy(), (M.part == 1).to_numpy(), (M.part == 2).to_numpy()
    print(f"samples: train {tr.sum():,} (daily), val {va.sum():,}, test {te.sum():,} (Fridays); built in {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(SEED)
    vi = np.where(va)[0]
    vi = np.sort(rng.choice(vi, min(len(vi), 80000), replace=False))
    tri = np.where(tr)[0]
    model = train(X[tri], M.iloc[tri], X[vi], M.iloc[vi])
    import torch
    os.makedirs(f"{RD}/results", exist_ok=True)
    torch.save(model.state_dict(), f"{RD}/results/chart_cnn2_seed{SEED}.pt")
    ev = np.where(va | te)[0]
    pu, pc = predict(model, X[ev])
    P = M.iloc[ev][["symbol", "date", "fwd5", "fwd20"]].rename(columns={"date": "week"}).assign(p_up=pu, p_crash=pc)
    P.to_parquet(f"{RD}/results/chart_cnn2_preds_seed{SEED}.parquet", index=False)
    for lab, sel in (("2022-23", P.week < VAL_END), ("2024+", P.week >= VAL_END)):
        Q = P[sel].copy()
        Q["ex5"] = Q.fwd5 - Q.groupby("week").fwd5.transform("median")
        Q["ex20"] = Q.fwd20 - Q.groupby("week").fwd20.transform("median")
        ic = Q.groupby("week").apply(lambda g: g.p_up.corr(g.ex5, method="spearman"))
        Q["crash_pct"] = Q.groupby("week").p_crash.rank(pct=True)
        Q["crashed"] = Q.groupby("week").fwd20.rank(pct=True) <= 0.10
        top = Q[Q.crash_pct >= 0.9]; rest = Q[Q.crash_pct < 0.9]
        print(f"{lab}: up-head IC {ic.mean():.4f} (t {ic.mean() / ic.std() * np.sqrt(ic.notna().sum()):.1f}); "
              f"crash-head top 10%: next-20-day excess {top.ex20.mean() * 100:+.2f}% vs rest {rest.ex20.mean() * 100:+.2f}%; "
              f"share of its top 10% that landed in the bottom 10%: {top.crashed.mean():.1%} (base 10%)", flush=True)
    print(f"::notice::chart CNN v2 seed {SEED}: done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
