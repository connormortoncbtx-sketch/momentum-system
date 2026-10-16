"""
research/stops_replay.py -- do the live stop rules rescue the volatile Apr-May picks? (Oct 16, 2026)
Replays each snapshot's top 10 (research/version_replay.py) on every logged week AFTER the
snapshot date, entry Monday open -> exit Friday close, with the live rules from
pipeline/04_model.py compute_suggested_stops:
  hard stop 7% (vol<10) / 10% (vol<20) / min(vol/2, 15)% ; activation max(3%, 0.75 x avg_win);
  then a trailing stop of min(vol/2, 15)% from the peak.
Path = daily opens and closes (Mon..Fri). A gap through a stop exits at the open; a close
through a stop exits at the stop price (assumes it filled intraday). 'pessimistic' also
stops out any untriggered position whose logged weekly LOW breached the hard stop.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "research")
import version_replay as V

DAYS = ["mon", "tue", "wed", "thu", "fri"]


def stop_return(r, pessimistic=False):
    vol, aw = r.weekly_vol_predicted, r.avg_win_magnitude
    e = r.mon_open
    if not (np.isfinite(e) and e > 0 and np.isfinite(r.fri_close)):
        return np.nan
    if not (np.isfinite(vol) and vol > 0):
        return r.fri_close / e - 1
    h = 7.0 if vol < 10 else 10.0 if vol < 20 else min(vol * 0.5, 15.0)
    a = max(3.0, (aw if np.isfinite(aw) else 0) * 0.75)
    t = min(vol * 0.5, 15.0)
    S, A = e * (1 - h / 100), e * (1 + a / 100)
    phase, peak = 1, e
    path = []
    for d in DAYS:
        for k in ("open", "close"):
            v = getattr(r, f"{d}_{k}", np.nan)
            if np.isfinite(v):
                path.append((k, v))
    for k, v in path[1:] if path and path[0][0] == "open" else path:
        if phase == 1:
            if v <= S:
                return (v if k == "open" else S) / e - 1
            if v >= A:
                phase, peak = 2, v
        else:
            peak = max(peak, v); T = peak * (1 - t / 100)
            if v <= T:
                return (v if k == "open" else T) / e - 1
    if pessimistic and phase == 1 and np.isfinite(r.weekly_low) and r.weekly_low <= S:
        return S / e - 1
    return r.fri_close / e - 1


def main():
    p = pd.read_csv("data/performance_log.csv", low_memory=False, parse_dates=["week_of"])
    p = p[p.sig_momentum_rs.notna() & p.regime.notna() & p.fri_close.notna() & p.mon_open.notna()]
    weeks = sorted(w for w, g in p.groupby("week_of") if len(g) > 500)
    rows = []
    for name, before in V.SNAPS.items():
        if before is None:
            continue
        c, feats, weights, model = V.snapshot(before)
        d0 = pd.Timestamp(before[:10])
        for wk in weeks:
            g = p[p.week_of == wk]
            top = V.score_week(g, feats, weights, model)
            if top is None or len(top) == 0:
                continue
            nostop = (top.fri_close / top.mon_open - 1).clip(-0.9, 3)
            st = top.apply(stop_return, axis=1); sp = top.apply(lambda r: stop_return(r, True), axis=1)
            rows.append(dict(snap=name, week=wk, oos=wk >= d0, vol=top.weekly_vol_predicted.median(),
                             nostop=nostop.mean(), stops=st.mean(), stops_pess=sp.mean(),
                             big10=(nostop >= 0.10).sum(), big_loss10=(nostop <= -0.10).sum(),
                             univ=(g.fri_close / g.mon_open - 1).clip(-0.9, 3).mean()))
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    o = d[d.oos]
    agg = o.groupby("snap", sort=False).agg(weeks=("week", "size"), picks_vol=("vol", "median"),
                                             big_winners=("big10", "sum"), big_losers=("big_loss10", "sum"),
                                             no_stops=("nostop", "mean"), with_stops=("stops", "mean"),
                                             with_stops_pess=("stops_pess", "mean"), universe=("univ", "mean"))
    for c in ("no_stops", "with_stops", "with_stops_pess", "universe"):
        agg[c] = agg[c] * 100
    cum = o.groupby("snap", sort=False).apply(lambda x: pd.Series({
        "cum_no_stops": ((1 + x.nostop).prod() - 1) * 100, "cum_stops": ((1 + x.stops).prod() - 1) * 100,
        "cum_stops_pess": ((1 + x.stops_pess).prod() - 1) * 100}))
    print("OUT OF SAMPLE (weeks after each snapshot), Mon open -> Fri close, mean % per week:")
    print(agg.join(cum).round(2).to_string())
    print("\nIN SAMPLE (weeks before each snapshot) for contrast:")
    i = d[~d.oos].groupby("snap", sort=False)[["nostop", "stops"]].mean() * 100
    print(i.round(2).to_string())


if __name__ == "__main__":
    main()
