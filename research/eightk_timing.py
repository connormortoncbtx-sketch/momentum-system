"""
research/eightk_timing.py -- are 8-Ks "too late"? (Nov 13, 2026; Connor). An 8-K is due within 4 business days of the
event, and most material news is press-released first. Measure, for the 59 large caps with 5-minute bars (2017-2026),
how much of each 8-K's price move happened BEFORE the filing was public (SEC acceptance time) vs after.
  pre  = excess return vs SPY from the close 3 trading days before the acceptance day to the last 5-minute close at or
         before acceptance (after-hours filings -> that day's close; pre-market -> prior close)
  post = excess return from acceptance to the close 3 trading days later (an investor reading the 8-K could start here)
By item group: median |pre|, median |post|, share of the total absolute move that came before the filing, and the
correlation of post with pre (positive = the move continues after the filing; negative = it reverses).
Large caps only -- small caps (where we trade contract wins) were measured separately in the LLM study at the first open
after acceptance. Public data only -> results/eightk_timing.csv.
"""
import glob
import os

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
GROUPS = {"2.02 earnings": "2.02", "1.01 material agreement": "1.01", "5.02 officer change": "5.02",
          "7.01 Reg FD": "7.01", "8.01 other": "8.01", "2.01 deal closed": "2.01", "2.05/2.06 restructuring": "2.05|2.06",
          "5.07 shareholder vote": "5.07"}


def main():
    bars = {}
    for f in glob.glob(f"{RD}/intraday_liquid/5m_*.parquet"):
        s = os.path.basename(f)[3:-8]
        d = pd.read_parquet(f, columns=["ts", "close"]).set_index("ts").close
        bars[s] = d[~d.index.duplicated()].sort_index()
    spy = bars.pop("SPY")
    for etf in ("QQQ", "IWM", "DIA") + tuple(k for k in list(bars) if k.startswith("XL")):
        bars.pop(etf, None)
    cols = ["tickers", "form", "items", "accepted_et"]
    F = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{RD}/sec/filings_20*.parquet"))])
    F = F[F.form.isin(["8-K"])].copy()
    F["t"] = F.tickers.fillna("").str.split(",").str[0].str.strip().str.upper()
    F = F[F.t.isin(bars) & F.accepted_et.notna() & (F.accepted_et >= "2017-01-10")]
    print(f"{len(F)} 8-Ks for {F.t.nunique()} large caps", flush=True)
    days = pd.DatetimeIndex(sorted(set(spy.index.normalize())))
    eod = spy.groupby(spy.index.normalize()).last()

    def px(series, when):
        j = series.index.searchsorted(when, side="right") - 1
        return series.iat[j] if j >= 0 else np.nan

    rows = []
    for r in F.itertuples():
        s = bars[r.t]; acc = pd.Timestamp(r.accepted_et)
        k = days.searchsorted(acc.normalize())
        if k < 3 or k + 3 >= len(days):
            continue
        t_pre = days[k - 3] + pd.Timedelta(hours=16)
        t_acc = min(acc, days[k] + pd.Timedelta(hours=16)) if k < len(days) and days[k] == acc.normalize() else acc
        t_post = days[k + 3] + pd.Timedelta(hours=16)
        a, b, c = px(s, t_pre), px(s, t_acc), px(s, t_post)
        ma, mb, mc = px(spy, t_pre), px(spy, t_acc), px(spy, t_post)
        if not all(np.isfinite([a, b, c, ma, mb, mc])) or min(a, b, c) <= 0:
            continue
        pre = (b / a - 1) - (mb / ma - 1); post = (c / b - 1) - (mc / mb - 1)
        if abs(pre) > 0.6 or abs(post) > 0.6:
            continue
        hr = acc.hour + acc.minute / 60
        when = "pre-market" if hr < 9.5 else ("market hours" if hr < 16 else "after close")
        items = str(r.items)
        grp = next((g for g, rx in GROUPS.items() if any(i in items for i in rx.split("|"))), "other")
        rows.append(dict(t=r.t, acc=acc, when=when, group=grp, pre=pre, post=post))
    D = pd.DataFrame(rows)

    def summ(x):
        tot = x.pre.abs() + x.post.abs()
        big = x[(x.pre + x.post).abs() > 0.04]
        return pd.Series(dict(n=len(x), med_abs_pre=x.pre.abs().median() * 100, med_abs_post=x.post.abs().median() * 100,
                              share_before=(x.pre.abs().sum() / tot.sum()) * 100, corr_pre_post=x.pre.corr(x.post),
                              big_moves=len(big), big_share_before=(big.pre.abs().sum() / (big.pre.abs() + big.post.abs()).sum() * 100) if len(big) else np.nan,
                              big_post_same_sign=(np.sign(big.pre) == np.sign(big.post)).mean() * 100 if len(big) else np.nan))
    T = D.groupby("group").apply(summ).sort_values("n", ascending=False)
    W = D.groupby("when").apply(summ)
    os.makedirs(f"{RD}/results", exist_ok=True)
    T.to_csv(f"{RD}/results/eightk_timing.csv")
    print("\nBY ITEM (excess vs SPY, %; 'share_before' = share of the absolute move done before the filing was public):")
    print(T.round(2).to_string())
    print("\nBY FILING TIME:\n" + W.round(2).to_string())
    # baseline: any random 3-day window has a 'pre' and 'post' too -- compare against non-filing days
    print("::notice::8-K timing: share of the move before the filing -- " +
          ", ".join(f"{g} {v:.0f}%" for g, v in T.share_before.items()))


if __name__ == "__main__":
    main()
