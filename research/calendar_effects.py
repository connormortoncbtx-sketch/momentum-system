"""
research/calendar_effects.py -- flow-driven calendar effects on the index (Nov 13, 2026). Mechanisms, not forecasts:
  TOM        turn of the month (last trading day + first 3): payroll / pension / 401k contributions buy at month start
             (Ariel 1987; Lakonishok & Smidt 1988; McConnell & Xu 2008)
  PREHOL     the trading day before a market holiday (Ariel 1990)
  MIDMONTH   trading days 9-15 of the month (the documented weak stretch, for contrast)
  OPEX_WK    the week of the 3rd-Friday options expiry
  MON / FRI  day-of-week
Each is scored as: return on effect days vs all other days (close-to-close), t-stat, and a long-only rule "in the ETF on
effect days, T-bills otherwise" vs buy-and-hold. ETFs: SPY, QQQ, IWM 2016-Oct 2026 (dev 2016-21 / holdout 2022-26);
plus the Fama-French daily US market (long history: dev 1963-2009 / holdout 2010+) if the file is daily.
Public data only -> results/calendar_effects.csv.
"""
import os

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)


def flags(idx):
    d = pd.DataFrame(index=idx)
    ym = idx.to_period("M")
    pos = pd.Series(range(len(idx)), index=idx)
    first = pos.groupby(ym).transform("min"); last = pos.groupby(ym).transform("max")
    k = pos - first + 1                                   # trading-day number in month
    d["TOM"] = (pos == last) | (k <= 3)
    d["MIDMONTH"] = (k >= 9) & (k <= 15)
    bd = pd.Series(idx[1:].append(pd.DatetimeIndex([idx[-1] + pd.Timedelta(days=1)])), index=idx)
    gap = np.busday_count(idx.values.astype("datetime64[D]"), bd.values.astype("datetime64[D]"))
    d["PREHOL"] = gap > 1                                 # next trading day skips a business day
    third_fri = idx.to_series().apply(lambda x: x + pd.offsets.WeekOfMonth(week=2, weekday=4) - pd.offsets.WeekOfMonth(week=2, weekday=4)
                                      if False else None)
    fri = pd.Series([pd.Timestamp(x.year, x.month, 1) + pd.offsets.WeekOfMonth(week=2, weekday=4) for x in idx], index=idx)
    fri = fri.where(fri.dt.month == idx.month, fri - pd.offsets.WeekOfMonth(week=2, weekday=4))
    d["OPEX_WK"] = (idx.isocalendar().week.values == pd.DatetimeIndex(fri.values).isocalendar().week.values) & \
                   (idx.year == pd.DatetimeIndex(fri.values).year)
    d["MON"] = idx.dayofweek == 0
    d["FRI"] = idx.dayofweek == 4
    return d


def score(r, F, rf, lab, split):
    out = []
    for part, m in (("dev", r.index < split), ("hold", r.index >= split)):
        rr, ff, rfx = r[m], F[m], rf.reindex(r.index[m]).fillna(0)
        bh = (1 + rr).prod() ** (252 / len(rr)) - 1
        bh_sh = rr.mean() / rr.std() * np.sqrt(252)
        for eff in F.columns:
            on = ff[eff].to_numpy()
            a, b = rr[on], rr[~on]
            se = np.sqrt(a.var() / len(a) + b.var() / len(b))
            strat = rr.where(on, rfx)
            out.append(dict(series=lab, part=part, effect=eff, days_pct=on.mean() * 100,
                            on_bp=a.mean() * 1e4, off_bp=b.mean() * 1e4, t=(a.mean() - b.mean()) / se,
                            share_of_total=(a.sum() / rr.sum() * 100) if rr.sum() > 0 else np.nan,
                            strat_CAGR=((1 + strat).prod() ** (252 / len(strat)) - 1) * 100,
                            strat_Sharpe=(strat - rfx).mean() / strat.std() * np.sqrt(252),
                            bh_CAGR=bh * 100, bh_Sharpe=bh_sh))
    return out


def main():
    e = pd.read_parquet(f"{RD}/etf/etf_daily.parquet")
    e["date"] = pd.to_datetime(e.date)
    rf = pd.Series(dtype=float)
    try:
        t = pd.read_parquet(f"{RD}/macro/fred_DTB3.parquet")
        t = t.set_index(t.columns[0]).iloc[:, 0]; t.index = pd.to_datetime(t.index)
        rf = (pd.to_numeric(t, errors="coerce") / 100 / 252).ffill()
    except Exception as ex:
        print("no T-bill:", ex)
    rows = []
    for s in ("SPY", "QQQ", "IWM"):
        c = e[e.symbol == s].set_index("date").close.sort_index()
        r = c.pct_change().dropna()
        rows += score(r, flags(r.index), rf.reindex(r.index, method="ffill") if len(rf) else r * 0, s, pd.Timestamp("2022-01-01"))
    try:
        ff = pd.read_parquet(f"{RD}/macro/french_factors.parquet")
        print("french_factors columns:", ff.columns.tolist()[:8], "rows", len(ff))
        dcol = [c for c in ff.columns if "date" in c.lower()]
        idx = pd.to_datetime(ff[dcol[0]]) if dcol else pd.to_datetime(ff.index)
        mk = [c for c in ff.columns if c.lower().replace("-", "") in ("mktrf", "mkt_rf")]
        rfc = [c for c in ff.columns if c.lower() == "rf"]
        if mk and pd.Series(idx).diff().dt.days.median() <= 3:
            scale = 100 if ff[mk[0]].abs().median() > 0.05 else 1
            rm = pd.Series((ff[mk[0]] + ff[rfc[0]]).to_numpy() / scale, index=idx).dropna()
            rff = pd.Series(ff[rfc[0]].to_numpy() / scale, index=idx)
            rm = rm[rm.index >= "1963-07-01"]
            rows += score(rm, flags(rm.index), rff, "FF_MKT", pd.Timestamp("2010-01-01"))
        else:
            print("french file is not daily -- long-history check skipped")
    except Exception as ex:
        print("french skipped:", ex)
    R = pd.DataFrame(rows)
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/calendar_effects.csv", index=False)
    print(R.round(2).to_string(index=False))
    tom = R[R.effect == "TOM"]
    print("::notice::TOM t-stats: " + ", ".join(f"{r.series}/{r.part} {r.t:+.1f}" for r in tom.itertuples()))


if __name__ == "__main__":
    main()
