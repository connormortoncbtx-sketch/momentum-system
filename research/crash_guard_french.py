"""
research/crash_guard_french.py -- momentum crash protection on 95 years of public data (Nov 15, 2026; Connor: "Sure
let's test" the Daniel & Moskowitz 2016 crash rule). Our own panel (2016+) has ~4 stress months, too few; earlier simple
crash guards on it were rejected (docs: "Stage 2", "Drawdown protection on the new books").

Data: Kenneth French Data Library (public, free): 10 portfolios formed on prior 12-2 month return (monthly + daily,
value- and equal-weighted) and the Fama-French market factor / T-bill rate (monthly + daily). Raw zips are cached under
research_data/french/ (public data).

The book that matters for us is LONG-ONLY: the top momentum decile ("Hi PRIOR"). The long-short UMD-style spread
(top - bottom decile) is reported for reference only (we cannot short at our size).

PRE-REGISTERED (expanding-window, no look-ahead; signals known at the end of month t-1, applied in month t):
  bear   cumulative market return over the past 24 months < 0
  mvol   annualised vol of daily market returns over the past 126 trading days
  R1  CRASH GUARD   in T-bills when bear AND mvol is above its expanding-window 2/3 quantile; otherwise the decile
  R2  HALF GUARD    same trigger, 50% decile / 50% T-bills
  R3  DM DYNAMIC    weight_t = mu_t / var_t, mu_t from an expanding regression of the book's excess return on
                    bear x mvol^2 (fit on months before t, >= 120 months), var_t = 126-day realised variance of the
                    book's daily returns; rescaled so the expanding-window mean weight is 1; clipped to [0, 2] (Reg T)
  PASS  for the long-only top decile (value- and equal-weighted): Sharpe up AND max drawdown smaller in BOTH halves,
        1937-1979 and 1980-2026 (first 10 years are burn-in). Also reported: 2017-2026.
Output: research_data/results/crash_guard_french.csv
"""
import io
import os
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {"mom_m": "10_Portfolios_Prior_12_2_CSV.zip", "mom_d": "10_Portfolios_Prior_12_2_Daily_CSV.zip",
         "ff_m": "F-F_Research_Data_Factors_CSV.zip", "ff_d": "F-F_Research_Data_Factors_daily_CSV.zip"}


def fetch(key):
    os.makedirs(f"{RD}/french", exist_ok=True)
    path = f"{RD}/french/{FILES[key]}"
    if not os.path.exists(path):
        req = urllib.request.Request(BASE + FILES[key], headers={"User-Agent": "Mozilla/5.0 momentum-research"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    z = zipfile.ZipFile(path)
    return z.read(z.namelist()[0]).decode("latin-1")


def blocks(text):
    """French CSV -> list of (title, DataFrame) for the monthly/daily blocks (index = date)."""
    out, title, cols, rows, prev = [], "", None, [], ""
    def flush():
        if cols and rows:
            df = pd.DataFrame(rows, columns=["d"] + cols)
            out.append((title, df))
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) > 1 and parts[0] == "" and any(re.search(r"[A-Za-z]", p) for p in parts[1:]):
            flush()
            title, cols, rows = prev, parts[1:], []
        elif cols and parts and re.fullmatch(r"\d{6}|\d{8}", parts[0]):
            rows.append([parts[0]] + [float(x) for x in parts[1:len(cols) + 1]])
        elif line.strip() and not (cols and parts and re.fullmatch(r"\d{4}", parts[0])):
            if cols and rows:
                flush(); cols, rows = None, []
            prev = line.strip()
    flush()
    res = []
    for t, df in out:
        n = len(df.d.iloc[0])
        df.index = pd.to_datetime(df.pop("d"), format="%Y%m%d" if n == 8 else "%Y%m")
        if n == 6:
            df.index = df.index.to_period("M")
        res.append((t, df.replace([-99.99, -999], np.nan) / 100.0))
    return res


def pick(bl, word):
    for t, df in bl:
        if word.lower() in t.lower():
            return df
    return bl[0][1]


def st(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    return dict(cagr=(eq.iloc[-1] ** (12 / len(r)) - 1) * 100, sharpe=r.mean() / r.std() * np.sqrt(12),
                maxdd=(eq / eq.cummax() - 1).min() * 100, worst12m=((1 + r).rolling(12).apply(np.prod, raw=True) - 1).min() * 100,
                n=len(r))


def main():
    mm, md = blocks(fetch("mom_m")), blocks(fetch("mom_d"))
    ffm, ffd = blocks(fetch("ff_m"))[0][1], blocks(fetch("ff_d"))[0][1]
    print("blocks:", [t[:50] for t, _ in mm][:4], flush=True)
    rf = ffm["RF"]
    mkt_m = ffm["Mkt-RF"] + rf
    mkt_d = ffd["Mkt-RF"] + ffd["RF"]
    mvol = (mkt_d.rolling(126).std() * np.sqrt(252)).groupby(mkt_d.index.to_period("M")).last()
    bear = ((1 + mkt_m).rolling(24).apply(np.prod, raw=True) - 1) < 0
    hi = mvol > mvol.expanding(60).quantile(2 / 3)
    trig = (bear & hi).shift(1).fillna(False).astype(bool)           # known at the end of t-1
    rows = []
    for wt, word in (("VW", "value weight"), ("EW", "equal weight")):
        M, Dd = pick(mm, word), pick(md, word)
        hi_c, lo_c = M.columns[-1], M.columns[0]
        books = {"top decile (long-only)": (M[hi_c], Dd[Dd.columns[-1]]),
                 "top - bottom (long-short, reference)": (M[hi_c] - M[lo_c] + rf, Dd[Dd.columns[-1]] - Dd[Dd.columns[0]])}
        for bname, (r, rd) in books.items():
            r = r.reindex(rf.index)
            ex = r - rf
            # R3 Daniel-Moskowitz dynamic weight
            bvar = (rd.rolling(126).var() * 21).groupby(rd.index.to_period("M")).last().reindex(rf.index).shift(1)
            x = (bear.astype(float) * mvol ** 2).shift(1).reindex(rf.index)
            w = pd.Series(np.nan, index=rf.index)
            idx = list(rf.index)
            for i, m in enumerate(idx):
                if i < 120:
                    continue
                h = pd.DataFrame({"y": ex.iloc[:i], "x": x.iloc[:i]}).dropna()
                if len(h) < 120:
                    continue
                b = np.polyfit(h.x, h.y, 1)
                mu = b[1] + b[0] * x.iloc[i] if np.isfinite(x.iloc[i]) else np.nan
                w.iloc[i] = mu / bvar.iloc[i] if bvar.iloc[i] > 0 else np.nan
            w = (w / w.expanding(60).mean().shift(1)).clip(0, 2)
            variants = {"base": r,
                        "R1 crash guard (cash)": r.where(~trig.reindex(rf.index).fillna(False), rf),
                        "R2 half guard": r.where(~trig.reindex(rf.index).fillna(False), 0.5 * r + 0.5 * rf),
                        "R3 DM dynamic": rf + w * ex}
            for lab, a, z in (("1937-1979", "1937-01", "1979-12"), ("1980-2026", "1980-01", "2026-12"),
                              ("2017-2026", "2017-01", "2026-12")):
                for vn, vr in variants.items():
                    s = st(vr.loc[a:z])
                    rows.append(dict(weight=wt, book=bname, period=lab, variant=vn, **s,
                                     months_guarded=int(trig.reindex(rf.index).loc[a:z].sum()),
                                     avg_dm_weight=float(w.loc[a:z].mean()) if vn.startswith("R3") else np.nan))
    R = pd.DataFrame(rows)
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/crash_guard_french.csv", index=False)
    print(R.round(2).to_string(index=False))
    print(f"::notice::crash guard French: data {rf.index.min()}..{rf.index.max()}, guarded months {int(trig.sum())}")


if __name__ == "__main__":
    main()
