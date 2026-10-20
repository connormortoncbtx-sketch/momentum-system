"""
research/kalshi/profit_lock.py -- should the weather MM bot lock in gains? (Oct 20, 2026)

Simulates the bot's book on the Kalshi weather trade sample: deep fills as in mm_perfill.py (a sweep that runs >= 2c
past its first price fills our 5-lot quote 2c behind), only 6-48h before close, net position capped at 25 per market.
Positions are marked hourly at each market's last trade and paid out at close. Account "day P&L" = equity now minus
equity at Central-time midnight -- what the app shows.
Rules compared with plain hold-to-settlement:
  TAKE X     once day P&L >= X% of capital, sell everything at the bid (mark -/+ 1c) paying the taker fee;
             the bot keeps quoting afterwards
  TAKE+STOP  same, then no new fills for the rest of the day
  TRAIL X    once day P&L has reached X%, sell everything if it falls back to half of its peak
Capital = 95th percentile of the book's cost at risk (what an account this size would need).
Dev < 2025-10-01 <= holdout.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from mm_perfill import load, SPLIT, OFF, SIZE

INV_CAP, MIN_H, MAX_H, HALF_SPREAD = 25, 6, 48, 0.01


def taker_fee(p):
    return np.ceil(0.07 * p * (1 - p) * 100 * 10 - 1e-9) / 100 / 10      # 10-lot rounding, per contract


def fills(t):
    g = t.sort_values(["ticker", "t"]).groupby(["ticker", "t", "taker_side"], sort=False)
    sw = g.agg(first=("p", "first"), R=("R", "first"), h=("h", "first"), close=("close_time", "first")).reset_index()
    t2 = t.merge(sw[["ticker", "t", "taker_side", "first"]], on=["ticker", "t", "taker_side"])
    beyond = np.where(t2.taker_side == "no", t2.p <= t2["first"] - OFF + 1e-9, t2.p >= t2["first"] + OFF - 1e-9)
    q = t2[beyond].groupby(["ticker", "t", "taker_side"]).c.sum().rename("qb")
    sw = sw.merge(q, on=["ticker", "t", "taker_side"])
    sw = sw[(sw.h >= MIN_H) & (sw.h <= MAX_H)]
    sw["px"] = np.where(sw.taker_side == "no", sw["first"] - OFF, sw["first"] + OFF)
    sw = sw[(sw.px >= 0.03) & (sw.px <= 0.97)]
    sw["q"] = np.where(sw.taker_side == "no", 1, -1) * np.minimum(sw.qb, SIZE)    # + = we bought YES
    out = []
    for tk, g in sw.sort_values("t").groupby("ticker"):               # net inventory cap
        pos, qs = 0, []
        for q in g.q.values:
            nq = int(np.clip(pos + q, -INV_CAP, INV_CAP) - pos); pos += nq; qs.append(nq)
        out.append(g.assign(q=qs))
    sw = pd.concat(out)
    return sw[sw.q != 0]


class Marks:
    """last trade price per ticker at or before an hour (sparse)."""
    def __init__(self, tt):
        self.d = {k: (g.t.values, g.p.values) for k, g in tt.sort_values("t")[["ticker", "t", "p"]].groupby("ticker")}

    def at(self, tk, h):
        ts, ps = self.d[tk]
        i = np.searchsorted(ts, np.datetime64(h.tz_convert(None)), side="right") - 1
        return ps[i] if i >= 0 else np.nan


def simulate(F, M, closes, R, hours, rule, X, cap):
    fh = {h: g for h, g in F.groupby("hr")}
    close_at = {h: list(v) for h, v in closes.groupby(closes).groups.items()}
    pos, cash = {}, 0.0
    day_start, cur_day, peak, stopped = 0.0, None, -1e9, False
    daily, n_lock = {}, 0

    def equity(h):
        e = cash
        for tk, q in pos.items():
            mk = M.at(tk, h)
            e += q * (mk if mk == mk else 0.5)
        return e
    for h in hours:
        d = h.tz_convert("America/Chicago").normalize()
        if d != cur_day:
            eq = equity(h)
            if cur_day is not None:
                daily[cur_day] = eq - day_start
            cur_day, day_start, peak, stopped = d, eq, -1e9, False
        if h in fh and not stopped:
            g = fh[h]
            for tk, q in g.groupby("ticker").q.sum().items():
                pos[tk] = pos.get(tk, 0) + q
            cash -= float((g.q * g.px).sum())
        for tk in close_at.get(h, []):
            cash += pos.pop(tk, 0) * R[tk]
        if rule == "hold" or not pos:
            continue
        dp = (equity(h) - day_start) / cap * 100
        peak = max(peak, dp)
        trig = (rule in ("take", "take+stop") and dp >= X) or (rule == "trail" and peak >= X and dp <= peak / 2)
        if trig:
            for tk, q in list(pos.items()):
                mk = M.at(tk, h); mk = 0.5 if mk != mk else mk
                cash += q * (mk - np.sign(q) * HALF_SPREAD) - abs(q) * taker_fee(min(max(mk, 0.01), 0.99))
            pos.clear(); n_lock += 1
            if rule in ("take+stop", "trail"):
                stopped = True
            if rule == "trail":
                peak = -1e9
    eq = equity(hours[-1])
    daily[cur_day] = eq - day_start
    return eq, pd.Series(daily), n_lock


def main():
    t = load()
    F = fills(t)
    F["hr"] = F.t.dt.ceil("h")
    pd.set_option("display.width", 220)
    rows = []
    for per, (a, b) in (("dev", (F.t.min(), SPLIT)), ("holdout", (SPLIT, F.t.max() + pd.Timedelta("1s")))):
        f = F[(F.t >= a) & (F.t < b)]
        tk = f.ticker.unique()
        tt = t[t.ticker.isin(tk)]
        hrs = pd.date_range(f.t.min().floor("h"), pd.to_datetime(f.close).max().ceil("h") + pd.Timedelta("1h"), freq="h")
        M = Marks(tt)
        closes = pd.to_datetime(f.groupby("ticker").close.first(), utc=True).dt.ceil("h")
        R = f.groupby("ticker").R.first()
        # capital: 95th pct of cost at risk of the held book
        cum = f.sort_values("t").assign(cost=lambda x: np.where(x.q > 0, x.q * x.px, -x.q * (1 - x.px)))
        risk = cum.set_index("hr").cost.groupby(level=0).sum().reindex(hrs, fill_value=0)
        settled = cum.groupby(closes.reindex(cum.ticker).values).cost.sum().reindex(hrs, fill_value=0)
        cap = float((risk.cumsum() - settled.cumsum()).quantile(0.95))
        base, bd, _ = simulate(f, M, closes, R, hrs, "hold", 0, cap)
        rows.append(dict(period=per, rule="hold to settlement", X=np.nan, pnl_pct_of_cap=base / cap * 100,
                         days=len(bd), day_sd_pct=bd.std() / cap * 100, sharpe_daily=bd.mean() / bd.std() * np.sqrt(365),
                         locks=0, capital=cap))
        for rule in ("take", "take+stop", "trail"):
            for X in (2, 4, 8, 12):
                e, dd, n = simulate(f, M, closes, R, hrs, rule, X, cap)
                rows.append(dict(period=per, rule=rule, X=X, pnl_pct_of_cap=e / cap * 100, days=len(dd),
                                 day_sd_pct=dd.std() / cap * 100, sharpe_daily=dd.mean() / dd.std() * np.sqrt(365),
                                 locks=n, capital=cap))
        print(per, "done", flush=True)
    T = pd.DataFrame(rows)
    os.makedirs(os.environ.get("RESEARCH_DATA", "research_data") + "/results", exist_ok=True)
    T.to_csv(os.environ.get("RESEARCH_DATA", "research_data") + "/results/kalshi_profit_lock.csv", index=False)
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
