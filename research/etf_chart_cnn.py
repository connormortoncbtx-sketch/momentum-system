"""
research/etf_chart_cnn.py -- chart-image CNN on market ETFs, for the ETF trend sleeve (Nov 14, 2026; Connor asked
whether the CNN can read regimes. A regime model has too little history to learn from (one market, a few regime
changes a decade); pooling ~85 ETFs across asset classes gives ~150k charts -- enough -- and asks the useful part of the
regime question: which markets are trending healthily and which are breaking down.)

Pool: the live ETFTREND sleeve's 41 ETFs + ~45 more (countries, industries, bond types, commodities, factors), daily
bars from Alpaca 2016 -> today (split/dividend adjusted).
Images (research/chart_images.py format): I20 = the last 20 trading days; I60 = the last 60 days drawn as 20 3-day
bars (open of the first day, high/low over the 3 days, close of the last, summed volume, 20-bar MA).
Label: beat the pool's median ETF over the next HORIZON trading days (20 default; 5 = next week, judged on a WEEKLY
rebalance of the same sleeve -- Connor, Nov 14; passed through research_run's `seed` input).
Split: train 2016-2021 (every trading day), early stopping on 2022-23, TEST 2024+; 2 seeds averaged per variant.
Evaluation on the sleeve's own 41 ETFs at each month-end, held one month, with the sleeve's sizing (10% vol target per
ETF, x4/N, capped at 1x), 2 bp per side:
  TREND        the live rule: in if the 63-day return > 0
  CNN          in if the CNN score is in the top half of the 41 that month
  TREND+VETO   the live rule, minus ETFs in the CNN's bottom third
  EQUAL        all 41 at the same sizing (reference)
Also the month-end rank IC of each variant's score with the next month's return. Public data only.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chart_images as CI  # noqa: E402

RD = os.environ.get("RESEARCH_DATA", "research_data")
SLEEVE = ['BITO', 'COPX', 'DBA', 'DBC', 'DIA', 'EEM', 'EFA', 'EWJ', 'FXB', 'FXE', 'FXI', 'FXY', 'GDX', 'GLD', 'HYG',
          'IBIT', 'IEF', 'IWM', 'LQD', 'QQQ', 'SHY', 'SLV', 'SMH', 'SPY', 'TIP', 'TLT', 'UNG', 'URA', 'USO', 'UUP',
          'VNQ', 'XBI', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
EXTRA = ['MDY', 'EWZ', 'EWG', 'EWU', 'EWC', 'EWA', 'EWT', 'EWY', 'INDA', 'EWW', 'VGK', 'KRE', 'XRT', 'XHB', 'ITB', 'IYT',
         'XME', 'OIH', 'GDXJ', 'SOXX', 'IBB', 'TAN', 'ICLN', 'KWEB', 'MUB', 'EMB', 'AGG', 'BND', 'FXA', 'CPER', 'PALL',
         'PPLT', 'WEAT', 'CORN', 'XOP', 'KBE', 'XLRE', 'XLC', 'IWF', 'IWD', 'MTUM', 'USMV', 'QUAL', 'EWH', 'EWS']
TRAIN_END, VAL_END = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")
SEEDS = int(os.environ.get("SEEDS") or 2)
EPOCHS = int(os.environ.get("EPOCHS") or 6)
HORIZON = int(os.environ.get("HORIZON") or os.environ.get("SEED") or 20)   # trading days ahead in the label (5 = weekly); research_run passes it via the seed input
REB = "W-FRI" if HORIZON <= 5 else "ME"                                     # weekly or month-end rebalance to match


def fetch():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    c = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    df = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=SLEEVE + EXTRA, timeframe=TimeFrame.Day,
                                           start=pd.Timestamp("2015-06-01"), adjustment=Adjustment.ALL, feed=DataFeed.SIP)).df
    df = df.reset_index()
    df["date"] = pd.to_datetime(df.timestamp).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return df[["symbol", "date", "open", "high", "low", "close", "volume"]].sort_values(["symbol", "date"])


def bars3(o, h, l, c, v):
    """aggregate the last 60 days into 20 3-day bars (most recent bar ends today)."""
    k = len(c) // 3 * 3
    o, h, l, c, v = o[-k:], h[-k:], l[-k:], c[-k:], v[-k:]
    r = lambda x: x.reshape(-1, 3)
    return r(o)[:, 0], r(h).max(1), r(l).min(1), r(c)[:, -1], r(v).sum(1)


def build(D):
    rows, I20, I60 = [], [], []
    for s, g in D.groupby("symbol", sort=False):
        dt_ = g.date.to_numpy(); o, h, l, c, v = (g[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "volume"))
        if len(c) < 140:
            continue
        ma20 = pd.Series(c).rolling(20).mean().to_numpy()
        for k in range(119, len(c)):
            s20 = slice(k - 19, k + 1)
            a = CI.draw(o[s20], h[s20], l[s20], c[s20], ma20[s20], v[s20])
            oo, hh, ll, cc, vv = bars3(o[k - 119:k + 1], h[k - 119:k + 1], l[k - 119:k + 1], c[k - 119:k + 1], v[k - 119:k + 1])
            ma3 = pd.Series(cc).rolling(20).mean().to_numpy()      # 20-bar MA of the 3-day closes (needs 40 bars = 120 days)
            b = CI.draw(oo[-20:], hh[-20:], ll[-20:], cc[-20:], ma3[-20:], vv[-20:])
            if a is None or b is None:
                continue
            fwd = c[k + HORIZON] / c[k] - 1 if k + HORIZON < len(c) else np.nan
            I20.append(a); I60.append(b)
            rows.append((s, pd.Timestamp(dt_[k]), fwd))
    M = pd.DataFrame(rows, columns=["symbol", "date", "fwd20"])
    M["y"] = (M.fwd20 > M.groupby("date").fwd20.transform("median")).astype(np.int64)
    return np.stack(I20), np.stack(I60), M


def fit_predict(X, M, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(os.cpu_count() or 4)
    tr = ((M.date < TRAIN_END) & M.fwd20.notna()).to_numpy()
    va = ((M.date >= TRAIN_END) & (M.date < VAL_END) & M.fwd20.notna()).to_numpy()
    model = CI.net(); opt = torch.optim.Adam(model.parameters(), lr=1e-4); ce = nn.CrossEntropyLoss()
    y = M.y.to_numpy(); ti, vi = np.where(tr)[0], np.where(va)[0]
    tv = lambda a: torch.from_numpy(a[:, None].astype(np.float32) / 255.0)
    best, state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train(); perm = np.random.permutation(ti)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = ce(model(tv(X[b])), torch.from_numpy(y[b])); opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for i in range(0, len(vi), 4096):
                b = vi[i:i + 4096]; vl += float(ce(model(tv(X[b])), torch.from_numpy(y[b]))) * len(b)
        vl /= len(vi)
        print(f"    seed {seed} epoch {ep + 1}: val loss {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    model.load_state_dict(state); model.eval()
    return CI.score([model], X)


def sleeve_book(px, sig_fn, label):
    """monthly book on the 41 sleeve ETFs with the live sizing; sig_fn(month_end_date) -> 0/1 Series."""
    ret = px.pct_change(); vol = ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    me = [px.index[px.index <= m][-1] for m in px.resample(REB).last().index if (px.index <= m).any()]
    me = sorted(set(me))
    out, prev = {}, pd.Series(dtype=float)
    for d0, d1 in zip(me[:-1], me[1:]):
        sig = sig_fn(d0).reindex(px.columns).fillna(0)
        tgt = (sig * (0.10 / vol.loc[d0]).clip(upper=2.0)).fillna(0)
        tgt = tgt / max((tgt != 0).sum(), 1) * 4
        if tgt.sum() > 1:
            tgt = tgt / tgt.sum()
        r = (px.loc[d1] / px.loc[d0] - 1).reindex(tgt.index).fillna(0)
        cost = (tgt - prev.reindex(tgt.index).fillna(0)).abs().sum() * 2 / 1e4
        out[d1] = float((tgt * r).sum()) - cost
        prev = tgt
    return pd.Series(out, name=label)


def stats(r):
    r = r.dropna()
    if len(r) < 6:
        return dict(CAGR=np.nan, Sharpe=np.nan, maxDD=np.nan)
    ppy = 52 if REB == "W-FRI" else 12
    eq = (1 + r).cumprod(); yrs = len(r) / ppy
    return dict(CAGR=(eq.iloc[-1] ** (1 / yrs) - 1) * 100, Sharpe=r.mean() / r.std() * np.sqrt(ppy),
                maxDD=(eq / eq.cummax() - 1).min() * 100)


def main():
    t0 = time.time()
    D = fetch()
    print(f"{D.symbol.nunique()} ETFs, {len(D):,} daily bars, {D.date.min().date()}..{D.date.max().date()}", flush=True)
    X20, X60, M = build(D)
    print(f"images: {len(M):,} (train {(M.date < TRAIN_END).sum():,}); {time.time() - t0:.0f}s", flush=True)
    for name, X in (("I20", X20), ("I60", X60)):
        M[name] = np.mean([fit_predict(X, M, s) for s in range(SEEDS)], axis=0)
        print(f"  {name} trained ({time.time() - t0:.0f}s)", flush=True)
    px = D.pivot_table(index="date", columns="symbol", values="close")[[s for s in SLEEVE if s in set(D.symbol)]]
    sc = {n: M.pivot_table(index="date", columns="symbol", values=n)[px.columns.intersection(M.symbol.unique())] for n in ("I20", "I60")}
    trend = lambda d: (px.loc[d] / px.shift(63).loc[d] - 1 > 0).astype(float)

    def cnn_top(n):
        return lambda d: (sc[n].loc[d].rank(pct=True) > 0.5).astype(float) if d in sc[n].index else pd.Series(dtype=float)

    def veto(n):
        return lambda d: trend(d) * ((sc[n].loc[d].rank(pct=True) > 1 / 3).astype(float).reindex(px.columns).fillna(1)
                                     if d in sc[n].index else 1)
    books = {"TREND (live rule)": trend, "EQUAL (all 41)": lambda d: px.loc[d].notna().astype(float),
             "CNN I20 top half": cnn_top("I20"), "CNN I60 top half": cnn_top("I60"),
             "TREND + I20 veto": veto("I20"), "TREND + I60 veto": veto("I60")}
    R = pd.concat([sleeve_book(px, f, k) for k, f in books.items()], axis=1)
    rows = []
    for lab, a, b in (("2022-23", TRAIN_END, VAL_END), ("2024+ TEST", VAL_END, pd.Timestamp("2100-01-01"))):
        x = R[(R.index >= a) & (R.index < b)]
        for k in R.columns:
            rows.append(dict(period=lab, book=k, **stats(x[k])))
    T = pd.DataFrame(rows)
    # month-end rank IC on the sleeve ETFs
    me = sorted(set(px.index[px.index <= m][-1] for m in px.resample(REB).last().index if (px.index <= m).any()))
    fwd = pd.DataFrame({d0: px.loc[d1] / px.loc[d0] - 1 for d0, d1 in zip(me[:-1], me[1:])}).T
    ics = []
    for n, s in (("I20", sc["I20"]), ("I60", sc["I60"]), ("63d trend", (px / px.shift(63) - 1))):
        ic = pd.Series({d: s.loc[d].corr(fwd.loc[d], method="spearman") for d in fwd.index if d in s.index})
        for lab, a, b in (("2022-23", TRAIN_END, VAL_END), ("2024+", VAL_END, pd.Timestamp("2100-01-01"))):
            x = ic[(ic.index >= a) & (ic.index < b)].dropna()
            ics.append(dict(score=n, period=lab, months=len(x), IC=x.mean(), t=x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan))
    os.makedirs(f"{RD}/results", exist_ok=True)
    T.to_csv(f"{RD}/results/etf_chart_cnn_h{HORIZON}.csv", index=False)
    M.drop(columns=["y"]).to_parquet(f"{RD}/results/etf_chart_cnn_h{HORIZON}_scores.parquet", index=False)
    pd.set_option("display.width", 200)
    print("\nSLEEVE BOOKS (41 ETFs, live sizing, monthly):\n" + T.round(2).to_string(index=False))
    print("\nMONTH-END RANK IC with the next month's return (41 sleeve ETFs):\n" + pd.DataFrame(ics).round(3).to_string(index=False))
    t = T[T.period == "2024+ TEST"].set_index("book")
    print(f"::notice::ETF chart CNN (label {HORIZON}d, rebalance {REB}) 2024+ Sharpe: " + "; ".join(f"{k} {v:.2f}" for k, v in t.Sharpe.items()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
