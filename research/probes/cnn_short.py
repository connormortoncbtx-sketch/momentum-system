"""
research/probes/cnn_short.py -- can the CNN's bottom decile actually be SHORTED? (Nov 14, 2026; Connor: "what if the
losing positions are shorted rather than vetoed").

1. Alpaca's asset flags (shortable, easy_to_borrow) for every stock the CNN put in its bottom 10% in 2024-26, and for
   the latest weeks' list. Caveat: the flags are TODAY's; a name easy to borrow now may not have been then, and names
   delisted since are not listed at all (counted as 'not shortable now').
2. Weekly short books on 2024-26 from Monday close, 5 trading days, with returns recomputed from daily bars WITHOUT the
   +150% clip (so squeezes are in): all bottom-decile names; only names flagged easy-to-borrow today; and a
   small-account version: the 20 lowest-scored easy-to-borrow names. Borrow fee scenarios 0 / 5 / 10 / 30% a year,
   plus spread cost on weekly turnover. Reported: CAGR, Sharpe, worst week, max drawdown, worst single-name week.
Read-only; no orders. Output: results/cnn_short.csv.
"""
import glob
import os

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
pd.set_option("display.width", 220)


def hs(adv):
    return np.select([adv >= 20e6, adv >= 5e6, adv >= 2e6, adv >= 1e6], [5, 10, 20, 35], 60).astype(float)


def main():
    P = pd.read_parquet(f"{RD}/results/chart_cnn_preds.parquet")
    P = P[P.week >= "2024-01-01"].copy()
    P["pct"] = P.groupby("week").cnn.rank(pct=True)
    Bt = P[P.pct <= 0.10].copy()
    from alpaca.trading.client import TradingClient
    tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    A = pd.DataFrame([dict(symbol=a.symbol, shortable=bool(a.shortable), etb=bool(a.easy_to_borrow), tradable=bool(a.tradable))
                      for a in tc.get_all_assets()]).drop_duplicates("symbol").set_index("symbol")
    Bt["shortable"] = Bt.symbol.map(A.shortable).fillna(False).astype(bool)
    Bt["etb"] = Bt.symbol.map(A.etb).fillna(False).astype(bool)
    last = Bt.week.max()
    L = Bt[Bt.week >= last - pd.Timedelta(weeks=7)]
    print(f"bottom-decile name-weeks 2024-26: {len(Bt):,} ({Bt.symbol.nunique():,} names). Flags today: shortable "
          f"{Bt.shortable.mean():.0%}, easy-to-borrow {Bt.etb.mean():.0%} of name-weeks")
    print(f"last 8 weeks ({L.week.min().date()}..{last.date()}): {L.symbol.nunique()} names; shortable {L.drop_duplicates('symbol').shortable.mean():.0%}, "
          f"easy-to-borrow {L.drop_duplicates('symbol').etb.mean():.0%}")
    # unclipped forward returns from daily bars
    D = pd.concat([pd.read_parquet(f, columns=["symbol", "date", "close", "volume"]) for f in sorted(glob.glob(f"{RD}/daily/ohlcv_202[3-6].parquet"))])
    C = D.pivot_table(index="date", columns="symbol", values="close")
    DV = (D.assign(dv=D.close * D.volume).pivot_table(index="date", columns="symbol", values="dv")).rolling(20, min_periods=10).mean()
    days = C.index
    k = days.searchsorted(pd.to_datetime(Bt.week).values, side="right") - 1   # index of the Friday (last trading day <= week)
    ok = (k + 6 < len(days))
    Bt = Bt[ok].copy(); k = k[ok]
    col = C.columns.get_indexer(Bt.symbol)
    good = col >= 0
    Bt, k, col = Bt[good], k[good], col[good]
    Cv = C.to_numpy()
    Bt["ret"] = Cv[k + 6, col] / Cv[k + 1, col] - 1                          # Monday close -> 5 trading days later
    Bt["adv"] = DV.to_numpy()[k, col]
    Bt = Bt.dropna(subset=["ret"])
    print(f"bottom-decile 5-day returns, unclipped: mean {Bt.ret.mean() * 100:+.2f}%, share >= +50%: {(Bt.ret >= 0.5).mean() * 100:.2f}%, "
          f"max {Bt.ret.max() * 100:+.0f}%")
    books = {"all bottom-decile names": Bt, "easy-to-borrow today only": Bt[Bt.etb],
             "20 lowest-scored easy-to-borrow (small account)": Bt[Bt.etb].sort_values("cnn").groupby("week").head(20)}
    rows = []
    for name, x in books.items():
        g = x.groupby("week")
        short = -g.ret.mean()
        cost = g.adv.apply(lambda a: hs(a.fillna(0).to_numpy()).mean() * 2 / 1e4)   # weekly full turnover, both legs
        worst_name = x.ret.max()
        for fee in (0.0, 0.05, 0.10, 0.30):
            r = (short - cost - fee / 52).dropna()
            eq = (1 + r).cumprod(); yrs = len(r) / 52
            rows.append(dict(book=name, names_per_week=round(g.size().median()), borrow_fee=f"{fee:.0%}",
                             CAGR=(eq.iloc[-1] ** (1 / yrs) - 1) * 100, Sharpe=r.mean() / r.std() * np.sqrt(52),
                             worst_week=r.min() * 100, maxDD=(eq / eq.cummax() - 1).min() * 100,
                             worst_single_name_week=worst_name * 100))
    R = pd.DataFrame(rows)
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/cnn_short.csv", index=False)
    print("\nSHORT BOOKS 2024-26 (weekly, squeezes included, spread costs; % figures):")
    print(R.round(2).to_string(index=False))
    s = R[(R.book.str.startswith("20 lowest")) & (R.borrow_fee == "0%")].iloc[0]
    print(f"::notice::CNN short: easy-to-borrow share {Bt.etb.mean():.0%}; small-account ETB book at 0% fee: CAGR {s.CAGR:+.1f}%, "
          f"Sharpe {s.Sharpe:.2f}, maxDD {s.maxDD:.0f}%, worst single-name week {s.worst_single_name_week:+.0f}%")


if __name__ == "__main__":
    main()
