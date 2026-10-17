"""
research/etf_momentum.py -- industry / theme ETF momentum: a new book? (Oct 17, 2026)

Idea (Connor): single stocks are noisy; theme ETFs ARE the theme. Industry momentum is a
documented effect (hot industries stay hot for months). Test it on every US-listed ETF that
is not leveraged / inverse / option-income / bond, with real liquidity.

Universe   nasdaqtrader symbol directories (ETF flag = Y), plus Alpaca INACTIVE fund-named
           assets to recover some delisted ETFs (survivorship: dead theme ETFs are under-
           represented anyway -- flatters results; noted).
Filter     name excludes leveraged/inverse/option-income/buffer/bond/money-market; point-in-time
           liquidity: 63-day median dollar volume >= $2M at each formation date; >= 260 days of history.
Signal     total return over L days (21, 63, 126, 252, 252 skipping the last 21) at the formation
           close; hold top N (5, 10, 20) equal weight from the NEXT close; rebalance weekly (Fri)
           or monthly; optional absolute filter (signal > 0 else that slot sits in cash);
           optional de-duplication (skip a candidate whose 126-day daily-return correlation with
           an already-picked ETF exceeds 0.9 -- avoids holding five semiconductor funds).
Costs      10 bps per unit of turnover.
Dev 2017-2023 chooses; holdout 2024-2026 scored once. Saves results/etf_momentum_daily.csv.
"""
import io, itertools, os, re, sys, time, urllib.request
import datetime as dt
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = f"{RD}/etf"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{RD}/results", exist_ok=True)
SPLIT, START = pd.Timestamp("2024-01-01"), pd.Timestamp("2017-01-01")
EXCL = re.compile(r"(\b(2X|3X|4X|1\.5X|2\.0X|-1X|-2X|-3X)\b|ULTRA|BULL\b|BEAR\b|INVERSE|LEVERAGED|\bSHORT\b|DAILY|"
                  r"YIELDMAX|OPTION INCOME|COVERED CALL|PREMIUM INCOME|BUFFER|DEFINED OUTCOME|HEDGED EQUITY|"
                  r"BOND|TREASURY|MUNI|FIXED INCOME|DURATION|CREDIT|FLOATING RATE|T-BILL|TBILL|MONEY MARKET|"
                  r"AGGREGATE|MORTGAGE|SENIOR LOAN|CLO\b|INCOME ETF|TARGET DATE|VOLATILITY|VIX|"
                  r"\bETN\b|NOTES?\b)", re.I)
FUNDNAME = re.compile(r"\b(ETF|FUND|TRUST|ISHARES|SPDR|INVESCO|VANGUARD|GLOBAL X|PROSHARES|DIREXION|VANECK|"
                      r"FIRST TRUST|WISDOMTREE|SCHWAB|ARK)\b", re.I)


def symdir():
    out = []
    for url, sym, etf in (("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol", "ETF"),
                          ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol", "ETF")):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
            d = pd.read_csv(io.BytesIO(raw), sep="|")
            d = d[(d[etf] == "Y") & (d.get("Test Issue", "N") == "N")]
            out.append(pd.DataFrame({"symbol": d[sym].astype(str), "name": d["Security Name"].astype(str)}))
        except Exception as e:
            print(f"::warning::{url}: {e}")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["symbol", "name"])


def alpaca_inactive_funds():
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus
    tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    rows = [{"symbol": a.symbol, "name": a.name or ""} for a in
            tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.INACTIVE))
            if a.name and FUNDNAME.search(a.name)]
    return pd.DataFrame(rows)


def universe():
    f = f"{OUT}/etf_universe.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    a = symdir(); a["active"] = True
    try:
        b = alpaca_inactive_funds(); b["active"] = False
    except Exception as e:
        print(f"::warning::alpaca inactive: {e}"); b = pd.DataFrame(columns=["symbol", "name", "active"])
    u = pd.concat([a, b], ignore_index=True).drop_duplicates("symbol")
    u = u[~u.symbol.str.contains(r"[\.\$ ]") & ~u.name.str.contains(EXCL)]
    u.to_parquet(f, index=False)
    print(f"ETF universe: {len(u):,} ({u.active.sum():,} active, {(~u.active).sum():,} inactive fund-named)")
    return u


def bars(symbols):
    f = f"{OUT}/etf_daily.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    c = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    parts = []
    syms = sorted(set(symbols) | {"SPY", "IWM"})
    for i in range(0, len(syms), 100):
        chunk = syms[i:i + 100]
        for k in range(4):
            try:
                d = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                                                      start=dt.datetime(2016, 1, 1), adjustment=Adjustment.ALL,
                                                      feed=DataFeed.SIP)).df.reset_index()
                parts.append(d[["symbol", "timestamp", "close", "volume"]]); break
            except Exception as e:
                print(f"  bars chunk {i}: {e}"); time.sleep(5 * (k + 1))
        if i % 1000 == 0:
            print(f"  bars {i:,}/{len(syms):,}", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d["date"] = pd.to_datetime(d.timestamp).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    d = d.drop(columns="timestamp")
    d.to_parquet(f, index=False)
    return d


def st(r):
    if len(r) < 20 or r.std() == 0:
        return dict(CAGR=np.nan, Sharpe=np.nan, DD=np.nan)
    eq = (1 + r).cumprod(); yrs = len(r) / 252
    return dict(CAGR=(eq.iloc[-1] ** (1 / yrs) - 1) * 100, Sharpe=r.mean() / r.std() * np.sqrt(252),
                DD=(eq / eq.cummax() - 1).min() * 100)


def main():
    u = universe()
    d = bars(list(u.symbol))
    C = d.pivot_table(index="date", columns="symbol", values="close").sort_index()
    V = d.pivot_table(index="date", columns="symbol", values="volume").sort_index().reindex_like(C)
    C = C[C.index >= "2016-01-01"]; V = V.reindex(C.index)
    R = C.pct_change(fill_method=None).clip(-0.5, 1.0)
    dv = (C * V).rolling(63, min_periods=40).median()
    age = C.notna().cumsum()
    elig = (dv >= 2e6) & (age >= 260) & ~C.columns.isin(["SPY", "IWM"])[None, :]
    print(f"symbols with bars: {C.shape[1]:,}; eligible per day (median): dev {int(elig[elig.index < SPLIT].sum(axis=1).median())}, "
          f"holdout {int(elig[elig.index >= SPLIT].sum(axis=1).median())}", flush=True)
    days = C.index[C.index >= START]
    fri = pd.DatetimeIndex(sorted(set(days[days.to_series().dt.weekday == 4])))
    mon_end = pd.DatetimeIndex([days[days <= m][-1] for m in pd.date_range(days[0], days[-1], freq="ME") if (days <= m).any()])
    sigs = {"21": C / C.shift(21) - 1, "63": C / C.shift(63) - 1, "126": C / C.shift(126) - 1,
            "252": C / C.shift(252) - 1, "252-21": C.shift(21) / C.shift(252) - 1}
    pos_of = {t: i for i, t in enumerate(C.index)}

    def run(L, N, reb, absf, dedup):
        sig = sigs[L]
        dates = fri if reb == "W" else mon_end
        W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
        prev = pd.Series(0.0, index=C.columns); cost = pd.Series(0.0, index=C.index)
        dl = list(dates)
        for j, t in enumerate(dl):
            s = sig.loc[t][elig.loc[t]].dropna().sort_values(ascending=False)
            picks = []
            if dedup:
                i0 = pos_of[t]; win = R.iloc[max(0, i0 - 125): i0 + 1]
                for sym in s.index:
                    if all(win[sym].corr(win[p]) < 0.9 for p in picks):
                        picks.append(sym)
                    if len(picks) == N:
                        break
            else:
                picks = list(s.index[:N])
            if absf:
                picks = [p for p in picks if s[p] > 0]
            tgt = pd.Series(0.0, index=C.columns); tgt[picks] = 1.0 / N
            i0 = pos_of[t]
            if i0 + 1 >= len(C.index):
                break
            start = C.index[i0 + 1]
            if j + 1 < len(dl) and pos_of[dl[j + 1]] + 1 < len(C.index):
                nxt = C.index[pos_of[dl[j + 1]] + 1]
                W.loc[(W.index > start) & (W.index <= nxt)] = tgt.values
            else:
                W.loc[W.index > start] = tgt.values
            cost[start] = (tgt - prev).abs().sum() * 10 / 1e4
            prev = tgt
        r = (W * R.fillna(0)).sum(axis=1) - cost
        return r[r.index >= START]

    grid = list(itertools.product(["21", "63", "126", "252", "252-21"], [5, 10, 20], ["W", "M"], [0, 1], [0, 1]))
    if os.environ.get("SMOKE"):
        grid = grid[::17]
    rows, series = [], {}
    for n, g in enumerate(grid):
        r = run(*g); series[g] = r
        d_, h_ = st(r[r.index < SPLIT]), st(r[r.index >= SPLIT])
        rows.append(dict(L=g[0], N=g[1], reb=g[2], absf=g[3], dedup=g[4], **{f"dev_{k}": v for k, v in d_.items()},
                         **{f"hold_{k}": v for k, v in h_.items()}))
        if n % 20 == 0:
            print(f"  grid {n}/{len(grid)}", flush=True)
    res = pd.DataFrame(rows).sort_values("dev_Sharpe", ascending=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    spy, iwm = R["SPY"].fillna(0), R["IWM"].fillna(0)
    print("\nBENCHMARKS")
    for nm, r in (("SPY", spy), ("IWM", iwm), ("all eligible ETFs, equal weight", (R.where(elig).mean(axis=1)).fillna(0))):
        r = r[r.index >= START]; a, b = st(r[r.index < SPLIT]), st(r[r.index >= SPLIT])
        print(f"  {nm:34s} dev {a['CAGR']:6.1f}%/yr Sh {a['Sharpe']:5.2f} DD {a['DD']:6.1f}% | holdout {b['CAGR']:6.1f}%/yr Sh {b['Sharpe']:5.2f} DD {b['DD']:6.1f}%")
    print("\nGRID (top 25 by DEV Sharpe)"); print(res.head(25).round(2).to_string(index=False))
    print(f"\ngrid medians: dev Sharpe {res.dev_Sharpe.median():.2f}, holdout Sharpe {res.hold_Sharpe.median():.2f}; "
          f"share of configs with holdout Sharpe > SPY's: {(res.hold_Sharpe > st(spy[spy.index >= SPLIT])['Sharpe']).mean() * 100:.0f}%")
    best = res.iloc[0]; g = (best.L, int(best.N), best.reb, int(best.absf), int(best.dedup))
    rb = series[g]
    dev = rb.index < SPLIT
    X = np.c_[np.ones(dev.sum()), iwm.reindex(rb.index)[dev], spy.reindex(rb.index)[dev]]
    b = np.linalg.lstsq(X, rb[dev].to_numpy(), rcond=None)[0]
    hedged = rb - b[1] * iwm.reindex(rb.index) - b[2] * spy.reindex(rb.index)
    bi = np.linalg.lstsq(np.c_[np.ones(dev.sum()), iwm.reindex(rb.index)[dev]], rb[dev].to_numpy(), rcond=None)[0]
    hedged_iwm = rb - bi[1] * iwm.reindex(rb.index)
    print(f"\nDEV-BEST {g}: beta IWM {b[1]:+.2f}, SPY {b[2]:+.2f}; alpha {b[0] * 252 * 100:+.1f}%/yr (dev)")
    for nm, r in (("raw", rb), ("hedged (IWM+SPY)", hedged), ("hedged (IWM only)", hedged_iwm)):
        a, h = st(r[dev]), st(r[~dev])
        print(f"  {nm:20s} dev {a['CAGR']:6.1f}%/yr Sh {a['Sharpe']:5.2f} DD {a['DD']:6.1f}% | holdout {h['CAGR']:6.1f}%/yr Sh {h['Sharpe']:5.2f} DD {h['DD']:6.1f}%")
    # what does it hold? last 3 formations of the dev-best (for the theme sanity check)
    last = rb.index[-1]
    print("\ncorrelation with SPY (daily): dev %.2f, holdout %.2f" % (rb[dev].corr(spy.reindex(rb.index)[dev]), rb[~dev].corr(spy.reindex(rb.index)[~dev])))
    pd.DataFrame({"etfmom": rb, "etfmom_h": hedged_iwm}).to_csv(f"{RD}/results/etf_momentum_daily.csv")
    res.to_csv(f"{RD}/results/etf_momentum_grid.csv", index=False)
    s = sigs[g[0]].loc[C.index[-1]][elig.loc[C.index[-1]]].dropna().sort_values(ascending=False).head(15)
    names = dict(zip(u.symbol, u.name))
    print("\nCURRENT TOP 15 by the dev-best signal:\n" + "\n".join(f"  {k:6s} {v * 100:+6.1f}%  {names.get(k, '')[:60]}" for k, v in s.items()))
    a, h = st(rb[dev]), st(rb[~dev]); hh = st(hedged_iwm[~dev])
    print(f"::notice::ETF momentum dev-best {g}: dev Sh {a['Sharpe']:.2f}, holdout Sh {h['Sharpe']:.2f} ({h['CAGR']:.0f}%/yr, DD {h['DD']:.0f}%); "
          f"IWM-hedged holdout Sh {hh['Sharpe']:.2f}; grid median holdout Sh {res.hold_Sharpe.median():.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
