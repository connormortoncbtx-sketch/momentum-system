"""
research/chart_cnn.py -- price-chart images read by a convolutional network (Jiang, Kelly & Xiu, "(Re-)Imag(in)ing
Price Trends", Journal of Finance 2023), Nov 13, 2026: "more sophisticated methods", step 2.

Images (the paper's 20-day format): 64 x 60 pixels, black and white. Each trading day is 3 pixels wide: open tick on
the left, high-low bar in the middle, close tick on the right; a 20-day moving-average line; volume bars along the
bottom 12 pixels. Prices scaled to the window's own range, so the network sees only SHAPE.
Samples: every Friday, every stock with price >= $5 and 20-day average dollar volume >= $2M (funds excluded).
Label: did the stock beat the median stock over the next 5 trading days, entered at the next day's close (Monday) --
the same timing the live system can trade.
Model: 3 conv blocks (5x3 kernels, batch norm, leaky ReLU, 2x1 max pool; 16/32/64 channels), dropout 0.5, 2 classes;
Adam, early stopping on validation loss. Ensemble of SEEDS models.
Split: train 2017-2021, validation 2022-2023 (early stopping only), TEST 2024 - latest (never touched in training).
Scoring on TEST: weekly rank IC of P(beat) with the next week's excess return; top / bottom decile; top-10 / top-50
weekly long books after spread costs vs the same universe's 4-week momentum, a 1-week reversal baseline, and SPY.
The 2024+ test still lacks most delisted stocks (fix pending) -- read it with that caveat.
Env: SEEDS (2), MAX_TRAIN (300000 images), EPOCHS (8), LIMIT (smoke test: symbols).
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RD = os.environ.get("RESEARCH_DATA", "research_data")
SEEDS = int(os.environ.get("SEEDS") or 2)
MAX_TRAIN = int(os.environ.get("MAX_TRAIN") or 300000)
EPOCHS = int(os.environ.get("EPOCHS") or 8)
LIMIT = int(os.environ.get("LIMIT") or 0)
KEEP_TRAIN = float(os.environ.get("KEEP_TRAIN") or 0.55)   # share of 2017-21 samples imaged (memory)
KEEP_VAL = float(os.environ.get("KEEP_VAL") or 0.2)
H, W, D = 64, 60, 20
PH, VH = 51, 12                   # price area rows, volume area rows (1 blank row between)
TRAIN_END, VAL_END = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")


def half_spread_bps(price, adv):
    t = np.select([adv >= 20e6, adv >= 5e6, adv >= 2e6], [5, 10, 20], 35).astype(float)
    return t


def load():
    D_ = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{RD}/daily/ohlcv_*.parquet"))], ignore_index=True)
    D_ = D_.dropna(subset=["open", "high", "low", "close"])
    D_ = D_[(D_.close > 0) & (D_.high >= D_.low)]
    try:
        import backtest as B
        p = B.load_panel(RD)
        funds = {s for s, isf in zip(p.symbols, p.is_fund) if isf}
        D_ = D_[~D_.symbol.isin(funds)]
    except Exception as e:
        print("fund filter skipped:", e)
    if LIMIT:
        keep = sorted(D_.symbol.unique())[:LIMIT]
        D_ = D_[D_.symbol.isin(keep)]
    return D_.sort_values(["symbol", "date"])


def build(Dd):
    """images, labels and meta for every (Friday, stock) sample."""
    days = np.array(sorted(Dd.date.unique()))
    fridays = pd.DatetimeIndex(days).to_series().groupby(pd.DatetimeIndex(days).to_period("W-FRI")).max().values
    imgs, meta = [], []
    rng = np.random.default_rng(1)
    t0 = time.time()
    for n, (sym, g) in enumerate(Dd.groupby("symbol", sort=False)):
        dt = g.date.to_numpy(); o = g.open.to_numpy(np.float64); hi = g.high.to_numpy(np.float64)
        lo = g.low.to_numpy(np.float64); c = g.close.to_numpy(np.float64); v = g.volume.to_numpy(np.float64)
        if len(c) < 60:
            continue
        ma20 = pd.Series(c).rolling(20).mean().to_numpy()
        dv20 = pd.Series(c * v).rolling(20).mean().to_numpy()
        pos = np.searchsorted(dt, fridays, side="right") - 1      # last trading day <= each Friday
        for fr, k in zip(fridays, pos):
            if k < 39 or k + 6 >= len(c) or dt[k] != fr:
                continue
            if c[k] < 5 or not dv20[k] >= 2e6:
                continue
            frt = pd.Timestamp(fr)
            if frt < pd.Timestamp("2017-01-01"):
                continue
            if frt < TRAIN_END and rng.random() > KEEP_TRAIN:      # thin training/validation samples to fit memory
                continue
            if TRAIN_END <= frt < VAL_END and rng.random() > KEEP_VAL:
                continue
            s = slice(k - 19, k + 1)
            oo, hh, ll, cc, mm, vv = o[s], hi[s], lo[s], c[s], ma20[s], v[s]
            top = max(hh.max(), np.nanmax(mm)); bot = min(ll.min(), np.nanmin(mm))
            if not top > bot:
                continue
            sc = lambda x: np.clip(((x - bot) / (top - bot) * (PH - 1)).round().astype(int), 0, PH - 1)
            img = np.zeros((H, W), np.uint8)
            yo, yh, yl, yc, ym = sc(oo), sc(hh), sc(ll), sc(cc), sc(mm)
            x0 = np.arange(D) * 3
            for d in range(D):                                   # rows from the top: price up = smaller row
                img[PH - 1 - yh[d]:PH - yl[d], x0[d] + 1] = 255
                img[PH - 1 - yo[d], x0[d]] = 255
                img[PH - 1 - yc[d], x0[d] + 2] = 255
            img[PH - 1 - ym, x0 + 1] = 255
            vm = vv.max()
            if vm > 0:
                vh = np.clip((vv / vm * VH).round().astype(int), 0, VH)
                for d in range(D):
                    if vh[d]:
                        img[H - vh[d]:H, x0[d] + 1] = 255
            ent, ex = c[k + 1], c[k + 6]                         # Monday close -> close 5 trading days later
            imgs.append(img)
            meta.append((sym, pd.Timestamp(fr), ex / ent - 1, c[k], dv20[k], c[k] / c[k - 20] - 1, c[k] / c[k - 5] - 1))
        if n % 500 == 0:
            print(f"  images: {n} symbols, {len(imgs):,} samples, {time.time() - t0:.0f}s", flush=True)
    M = pd.DataFrame(meta, columns=["symbol", "week", "fwd", "price", "dv20", "r20", "r5"])
    M["fwd"] = M.fwd.where(M.fwd.abs() < 1.5)
    M["ex"] = M.fwd - M.groupby("week").fwd.transform("median")
    M["y"] = (M.ex > 0).astype(np.int64)
    return np.stack(imgs), M


def net():
    import torch.nn as nn

    def blk(i, o):
        return [nn.Conv2d(i, o, (5, 3), padding=(2, 1)), nn.BatchNorm2d(o), nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1))]
    return nn.Sequential(*blk(1, 16), *blk(16, 32), *blk(32, 64), nn.Flatten(), nn.Dropout(0.5), nn.Linear(64 * 8 * 60, 2))


def train(X, y, Xv, yv, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(os.cpu_count() or 4)
    model = net(); opt = torch.optim.Adam(model.parameters(), lr=1e-4); lossf = nn.CrossEntropyLoss()
    best, best_state, bad = 1e9, None, 0
    tv = lambda a: torch.from_numpy(a[:, None].astype(np.float32) / 255.0)
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); perm = np.random.permutation(len(X)); tot = 0.0
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]
            opt.zero_grad(); loss = lossf(model(tv(X[b])), torch.from_numpy(y[b])); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for i in range(0, len(Xv), 2048):
                vl += float(lossf(model(tv(Xv[i:i + 2048])), torch.from_numpy(yv[i:i + 2048]))) * len(Xv[i:i + 2048])
        vl /= len(Xv)
        print(f"  seed {seed} epoch {ep + 1}: train loss {tot / len(X):.4f}, val loss {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    model.load_state_dict(best_state); model.eval()
    return model


def predict(models, X):
    import torch
    out = np.zeros(len(X))
    with torch.no_grad():
        for m_ in models:
            for i in range(0, len(X), 4096):
                xb = torch.from_numpy(X[i:i + 4096][:, None].astype(np.float32) / 255.0)
                out[i:i + 4096] += torch.softmax(m_(xb), 1)[:, 1].numpy()
    return out / len(models)


def book(M, col, n):
    """weekly top-n equal-weight long book, after spread costs on turnover."""
    res, prev = {}, set()
    for wk, g in M.groupby("week"):
        g = g.dropna(subset=["fwd", col])
        if len(g) < n * 3:
            continue
        top = g.nlargest(n, col)
        cur = set(top.symbol)
        cost = half_spread_bps(top.price.to_numpy(), top.dv20.to_numpy()).mean() / 1e4 * (len(cur - prev) + len(prev - cur)) / n
        res[wk] = top.fwd.mean() - cost
        prev = cur
    r = pd.Series(res)
    eq = (1 + r).cumprod(); yrs = len(r) / 52
    return dict(CAGR=(eq.iloc[-1] ** (1 / yrs) - 1) * 100, Sharpe=r.mean() / r.std() * np.sqrt(52),
                maxDD=(eq / eq.cummax() - 1).min() * 100, weeks=len(r))


def main():
    t0 = time.time()
    Dd = load()
    print(f"daily rows {len(Dd):,}, symbols {Dd.symbol.nunique()}", flush=True)
    X, M = build(Dd)
    del Dd
    ok = M.fwd.notna().to_numpy()
    X, M = X[ok], M[ok].reset_index(drop=True)
    tr = (M.week < TRAIN_END).to_numpy() & (M.week >= "2017-01-01").to_numpy()
    va = ((M.week >= TRAIN_END) & (M.week < VAL_END)).to_numpy()
    te = (M.week >= VAL_END).to_numpy()
    print(f"samples: train {tr.sum():,}, val {va.sum():,}, test {te.sum():,}; images built in {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    tri = np.where(tr)[0]
    if len(tri) > MAX_TRAIN:
        tri = np.sort(rng.choice(tri, MAX_TRAIN, replace=False))
    vai = np.where(va)[0]
    if len(vai) > 60000:
        vai = np.sort(rng.choice(vai, 60000, replace=False))
    y = M.y.to_numpy()
    models = [train(X[tri], y[tri], X[vai], y[vai], s) for s in range(SEEDS)]
    M["cnn"] = predict(models, X)
    M["mom4"] = M.r20; M["rev1"] = -M.r5
    rows = []
    for lab, sel in (("validation 2022-23", va), ("TEST 2024+", te)):
        T = M[sel]
        for col in ("cnn", "mom4", "rev1"):
            ic = T.groupby("week").apply(lambda g: g[col].corr(g.ex, method="spearman"))
            dec = T.groupby("week").apply(lambda g: g[g[col].rank(pct=True) >= 0.9].ex.mean() - g[g[col].rank(pct=True) <= 0.1].ex.mean())
            topd = T.groupby("week").apply(lambda g: g[g[col].rank(pct=True) >= 0.9].ex.mean())
            b10, b50 = book(T, col, 10), book(T, col, 50)
            rows.append(dict(period=lab, signal=col, weeks=ic.notna().sum(), IC=ic.mean(), IC_t=ic.mean() / ic.std() * np.sqrt(ic.notna().sum()),
                             top_minus_bottom_bp=dec.mean() * 1e4, top_decile_vs_median_bp=topd.mean() * 1e4,
                             top10_CAGR=b10["CAGR"], top10_Sharpe=b10["Sharpe"], top10_DD=b10["maxDD"],
                             top50_CAGR=b50["CAGR"], top50_Sharpe=b50["Sharpe"], top50_DD=b50["maxDD"]))
    R = pd.DataFrame(rows)
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/chart_cnn.csv", index=False)
    M[["symbol", "week", "cnn", "fwd", "ex"]].to_parquet(f"{RD}/results/chart_cnn_preds.parquet", index=False)
    pd.set_option("display.width", 250)
    print("\n" + R.round(3).to_string(index=False))
    t = R[(R.period == "TEST 2024+") & (R.signal == "cnn")].iloc[0]
    print(f"::notice::chart CNN TEST 2024+: IC {t.IC:.4f} (t {t.IC_t:.1f}), top-bottom decile {t.top_minus_bottom_bp:.0f} bp/wk, "
          f"top-10 {t.top10_CAGR:+.1f}%/Sh {t.top10_Sharpe:.2f}, top-50 {t.top50_CAGR:+.1f}%/Sh {t.top50_Sharpe:.2f}; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
