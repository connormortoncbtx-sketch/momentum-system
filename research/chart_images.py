"""
research/chart_images.py -- the chart-image format and network of research/chart_cnn.py, factored out so live
scoring (automation/shadow_fip_veto.py) draws EXACTLY the images the model was trained on (Nov 14, 2026).
Keep in sync with chart_cnn.py (same constants, same drawing, same layers).
"""
import numpy as np

H, W, D = 64, 60, 20
PH, VH = 51, 12


def draw(oo, hh, ll, cc, mm, vv):
    """64x60 uint8 image from the last 20 days' open/high/low/close, 20-day MA and volume (adjusted prices).
    Returns None when the window has no range."""
    top = max(hh.max(), np.nanmax(mm)); bot = min(ll.min(), np.nanmin(mm))
    if not top > bot:
        return None
    sc = lambda x: np.clip(((x - bot) / (top - bot) * (PH - 1)).round().astype(int), 0, PH - 1)
    img = np.zeros((H, W), np.uint8)
    yo, yh, yl, yc, ym = sc(oo), sc(hh), sc(ll), sc(cc), sc(mm)
    x0 = np.arange(D) * 3
    for d in range(D):
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
    return img


def net():
    import torch.nn as nn

    def blk(i, o):
        return [nn.Conv2d(i, o, (5, 3), padding=(2, 1)), nn.BatchNorm2d(o), nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1))]
    return nn.Sequential(*blk(1, 16), *blk(16, 32), *blk(32, 64), nn.Flatten(), nn.Dropout(0.5), nn.Linear(64 * 8 * 60, 2))


def load_models(paths):
    import torch
    ms = []
    for p in paths:
        m = net(); m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); ms.append(m)
    return ms


def score(models, X):
    """P(beat the median next week), averaged over the ensemble. X: (n, 64, 60) uint8."""
    import torch
    out = np.zeros(len(X))
    with torch.no_grad():
        for m in models:
            for i in range(0, len(X), 4096):
                xb = torch.from_numpy(X[i:i + 4096][:, None].astype(np.float32) / 255.0)
                out[i:i + 4096] += torch.softmax(m(xb), 1)[:, 1].numpy()
    return out / len(models)


def images_at(daily, day):
    """Images for every symbol at trading day `day` from a long daily frame (symbol, date, open, high, low, close,
    volume; adjusted). Same filters as chart_cnn.build(): price >= $5, 20-day $ volume >= $2M, >= 40 days of bars."""
    import pandas as pd
    out, syms = [], []
    day = pd.Timestamp(day)
    for s, g in daily[daily.date <= day].groupby("symbol", sort=False):
        g = g.sort_values("date")
        if len(g) < 40 or g.date.iloc[-1] != day:
            continue
        c = g.close.to_numpy(np.float64)
        dv20 = float(np.mean(c[-20:] * g.volume.to_numpy(np.float64)[-20:]))
        if c[-1] < 5 or not dv20 >= 2e6:
            continue
        ma = pd.Series(c).rolling(20).mean().to_numpy()[-20:]
        img = draw(g.open.to_numpy(np.float64)[-20:], g.high.to_numpy(np.float64)[-20:], g.low.to_numpy(np.float64)[-20:],
                   c[-20:], ma, g.volume.to_numpy(np.float64)[-20:])
        if img is not None:
            out.append(img); syms.append(s)
    return (np.stack(out) if out else np.zeros((0, H, W), np.uint8)), syms
