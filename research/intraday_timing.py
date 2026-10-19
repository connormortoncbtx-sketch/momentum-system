"""
research/intraday_timing.py -- two intraday follow-ups (Oct 19, 2026). Dev 2017-2023 picks, holdout 2024-2026.

1. MULTI-TIMEFRAME TREND ("macro-micro"): on the 62 liquid symbols (cache from intraday_strats.py), go long when the
   60-minute trend is up by > a, the prior 15 minutes dipped by > d, and the current 5-minute bar turned up; optionally
   only if the symbol's 20-day return is positive (daily filter). Mirror for shorts in the long+short variant.
   Enter at the next bar's open, hold 30 / 60 minutes or to the close; first signal per symbol per day only.
   Costs per side 1 bp (index ETFs) / 2 bp (others).

2. EXECUTION TIMING for the overnight book: the live-proxy top-10 picks (research/intraday_picks.csv, 5-minute bars in
   intraday_5m_<year>.parquet), weekday nights. Baseline = buy at the close (15:55 bar close ~ closing auction) and
   sell at the open (09:30 bar open ~ opening auction). Alternatives move the entry earlier (15:30, 15:45, TWAP of the
   last 30 min) or the exit later (09:35 ... 10:30, TWAP of the first 30 min, and conditional exits: hold to 10:00
   only if the stock gapped up / only if the first 5-minute bar was up). Leaving an auction means paying the spread:
   the improvement must beat a half-spread of 3 / 5 / 10 bp per moved leg for these mid/small caps.
"""
import itertools, os
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
SPLIT = pd.Timestamp("2024-01-01")
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{RD}/results", exist_ok=True)


# ---------------------------------------------------------------------------------------------- 1. multi-timeframe
def mtf():
    import intraday_strats as S
    O, H, L, C = S.load()
    date = pd.Series(C.index.normalize(), index=C.index)
    tod = pd.Series(C.index.time, index=C.index)
    win = (tod >= pd.Timestamp("10:30").time()) & (tod <= pd.Timestamp("15:00").time())
    dclose = C.groupby(C.index.normalize()).last()
    mom20 = (dclose / dclose.shift(20) - 1).shift(1)                    # known before the day starts
    mom20_b = mom20.reindex(date.values).set_index(C.index)
    tr60 = C / C.shift(12) - 1
    dip15 = C.shift(1) / C.shift(4) - 1
    up = C > C.shift(1); dn = C < C.shift(1)
    same12 = date == date.shift(12)
    last_c = C.groupby(C.index.normalize()).transform("last")
    rows = []
    for a, d, hold, dfilt, both in itertools.product((0.002, 0.004), (0.0015, 0.003), (6, 12, "close"), (False, True), (False, True)):
        exitp = last_c if hold == "close" else C.shift(-hold)
        ok_h = True if hold == "close" else (date == date.shift(-hold))
        recs = []
        for side in ((1, -1) if both else (1,)):
            if side == 1:
                sig = (tr60 > a) & (dip15 < -d) & up
                if dfilt:
                    sig &= mom20_b > 0
            else:
                sig = (tr60 < -a) & (dip15 > d) & dn
                if dfilt:
                    sig &= mom20_b < 0
            mask = (same12 & ok_h & win).astype(bool)
            sig = sig.fillna(False).astype(bool).mul(mask, axis=0).astype(bool)
            entry = O.shift(-1)
            r = side * (exitp / entry - 1)
            st = sig.stack()
            st = st[st]
            if not len(st):
                continue
            df = st.reset_index()
            df.columns = ["ts", "sym", "_"]
            df["date"] = df.ts.dt.normalize()
            df = df.drop_duplicates(["sym", "date"])                      # first signal per symbol per day
            g = r.stack().reindex(pd.MultiIndex.from_arrays([df.ts, df.sym])).values
            df["gross"] = g
            df["net"] = df.gross - 2 * df.sym.map(S.COST)
            recs.append(df.dropna(subset=["gross"]))
        if not recs:
            continue
        tr = pd.concat(recs)
        rows += S.split_stats(tr[["date", "sym", "gross", "net"]], "MTF trend+dip",
                              f"60m>{a*100:.1f}% dip15>{d*100:.2f}% hold {hold} daily{'+' if dfilt else '-'} {'L/S' if both else 'long'}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- 2. overnight timing
def timing():
    picks = pd.read_csv(os.path.join(HERE, "intraday_picks.csv"), parse_dates=["formation_week", "trade_week"])
    picks = picks[picks["rank"] <= 10]
    parts = []
    for y in range(2017, 2027):
        p = f"{RD}/intraday_5m_{y}.parquet"
        if os.path.exists(p):
            parts.append(pd.read_parquet(p))
    b = pd.concat(parts, ignore_index=True)
    b["trade_week"] = pd.to_datetime(b.trade_week)
    b = b.merge(picks[["trade_week", "symbol"]], on=["trade_week", "symbol"])
    b["date"] = b.ts.dt.normalize(); b["t"] = b.ts.dt.strftime("%H:%M")

    def at_close(t):        # close of the bar that ENDS at time t (bar start = t - 5 min)
        st = (pd.Timestamp("2000-01-01 " + t) - pd.Timedelta(minutes=5)).strftime("%H:%M")
        return b[b.t == st].set_index(["trade_week", "symbol", "date"]).c
    px = pd.DataFrame({
        "open": b[b.t == "09:30"].set_index(["trade_week", "symbol", "date"]).o,
        "c0935": at_close("09:35"), "c0940": at_close("09:40"), "c0945": at_close("09:45"),
        "c1000": at_close("10:00"), "c1030": at_close("10:30"),
        "c1530": at_close("15:30"), "c1545": at_close("15:45"), "close": at_close("16:00"),
    })
    first30 = b[(b.t >= "09:30") & (b.t < "10:00")].groupby(["trade_week", "symbol", "date"]).c.mean()
    last30 = b[(b.t >= "15:30") & (b.t <= "15:55")].groupby(["trade_week", "symbol", "date"]).c.mean()
    px["twap_first30"], px["twap_last30"] = first30, last30
    px = px.reset_index().sort_values(["trade_week", "symbol", "date"])
    nx = px.groupby(["trade_week", "symbol"]).shift(-1)                   # next trading day of the same pick-week
    ok = nx.date.notna() & ((nx.date - px.date).dt.days <= 4)
    px, nx = px[ok], nx[ok]
    entries = {"close (auction)": px.close, "15:30": px.c1530, "15:45": px.c1545, "TWAP last 30m": px.twap_last30}
    gap_up = nx.open > px.close
    first_up = nx.c0935 > nx.open
    exits = {"open (auction)": nx.open, "09:35": nx.c0935, "09:40": nx.c0940, "09:45": nx.c0945, "10:00": nx.c1000,
             "10:30": nx.c1030, "TWAP first 30m": nx.twap_first30,
             "10:00 if gapped up else open": np.where(gap_up, nx.c1000, nx.open),
             "10:00 if first bar up else 09:35": np.where(first_up, nx.c1000, nx.c0935)}
    base = (nx.open / px.close - 1).values
    rows = []
    for (en, e), (xn, x) in itertools.product(entries.items(), exits.items()):
        r = np.asarray(x, float) / np.asarray(e, float) - 1
        moved = (en != "close (auction)") + (xn != "open (auction)")
        d = pd.DataFrame({"night": px.date.values, "r": r, "base": base}).dropna()
        n = d.groupby("night")[["r", "base"]].mean()                       # equal-weight book per night
        n["diff"] = n.r - n.base
        for per, m in (("dev", n.index < SPLIT), ("hold", n.index >= SPLIT)):
            q = n[m]
            if len(q) < 30:
                continue
            se = q["diff"].std() / np.sqrt(len(q))
            rows.append(dict(entry=en, exit=xn, period=per, nights=len(q), night_bp=q.r.mean() * 1e4, base_bp=q.base.mean() * 1e4,
                             improve_bp=q["diff"].mean() * 1e4, t=q["diff"].mean() / se if se else np.nan, legs_moved=moved,
                             **{f"net_improve_hs{hs}": q["diff"].mean() * 1e4 - moved * hs for hs in (3, 5, 10)}))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    T = timing()
    T.to_csv(f"{RD}/results/overnight_timing.csv", index=False)
    W = T.pivot_table(index=["entry", "exit"], columns="period", values=["improve_bp", "t", "net_improve_hs5"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    print("OVERNIGHT EXECUTION TIMING vs close->open auctions (bp per night, equal-weight top-10; t paired by night)")
    print(W.round(2).sort_values("improve_bp_dev", ascending=False).to_string())
    d = T[(T.period == "dev") & (T.legs_moved > 0)].sort_values("net_improve_hs5", ascending=False).iloc[0]
    h = T[(T.period == "hold") & (T.entry == d.entry) & (T.exit == d.exit)].iloc[0]
    print(f"::notice::overnight timing: dev pick [{d.entry} -> {d.exit}] improve {d.improve_bp:+.1f}bp (t {d.t:.1f}), net of 5bp half-spread "
          f"{d.net_improve_hs5:+.1f}bp -> holdout improve {h.improve_bp:+.1f}bp (t {h.t:.1f}), net {h.net_improve_hs5:+.1f}bp")

    M = mtf()
    M.to_csv(f"{RD}/results/intraday_mtf.csv", index=False)
    W = M.pivot_table(index="variant", columns="period", values=["trades_per_day", "gross_bp", "net_bp", "sharpe"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    print("\nMULTI-TIMEFRAME TREND + DIP")
    print(W.round(2).sort_values("sharpe_dev", ascending=False).to_string())
    d = M[(M.period == "dev") & (M.trades >= 250)].sort_values("sharpe", ascending=False).iloc[0]
    h = M[(M.period == "hold") & (M.variant == d.variant)]
    h = h.iloc[0] if len(h) else None
    print(f"::notice::MTF: dev pick [{d.variant}] dev Sharpe {d.sharpe:.2f} net {d.net_bp:+.1f}bp -> "
          + (f"holdout Sharpe {h.sharpe:.2f}, net {h.net_bp:+.1f}bp (gross {h.gross_bp:+.1f}), {h.trades_per_day:.1f} trades/day" if h is not None else "no holdout"))


if __name__ == "__main__":
    main()
