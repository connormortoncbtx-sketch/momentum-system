"""
research/macro_signal.py -- the pre-registered pass-3c rule as a reusable function:
"S5 vol-managed + ridge on/off" from research/macro_regime.py.

    weight_t = clip(target_vol / umd_vol6_t, 0, 2) * 1[ridge prediction_t > 0]

decided at month end t, applied to month t+1. Long-only use caps it at 1 and
holds the remainder in SPY.

french_lag: months of delay on Ken French data (it posts about a month late).
With lag L, French-derived features use data through t-L, and the ridge only
trains on rows whose outcome month is <= t-L. french_lag=0 reproduces the
research exactly.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_VOL_DEV = ("1972-01-31", "2004-12-31")
RIDGE_ALPHA = 12.0


def build(macro_dir, french_lag=0, end=None):
    D = Path(macro_dir)
    rd = lambda n: pd.read_parquet(D / f"{n}.parquet").set_index("date")["value"].sort_index()
    fac = pd.read_parquet(D / "french_factors.parquet").set_index("date")
    mom = pd.read_parquet(D / "french_momentum.parquet").set_index("date")["Mom"]
    end = pd.Timestamp(end) if end is not None else max(rd("fred_GS10").index.max(), mom.index.max())
    months = pd.date_range("1960-01-31", end + pd.offsets.MonthEnd(0), freq="ME")

    def month_end_last(s, lag_days=0):
        s = s.copy(); s.index = s.index + pd.Timedelta(days=lag_days)
        return s.resample("ME").last().reindex(months).ffill(limit=2)

    def monthly_obs(s, lag_months=0):
        # extend to the full month grid BEFORE lagging (else the newest value falls off
        # the end), then carry one missing month (e.g. the Oct-2025 shutdown gap)
        return s.resample("ME").last().reindex(months).shift(lag_months).ffill(limit=1)

    umd_m = (1 + mom).resample("ME").prod().reindex(months) - 1
    umd_m[months > mom.index.max()] = np.nan
    mkt_d = fac["Mkt-RF"] + fac["RF"]
    mkt_m = (1 + mkt_d).resample("ME").prod().reindex(months) - 1
    mkt_m[months > mkt_d.index.max()] = np.nan

    X = pd.DataFrame(index=months)
    gs10, tb3, ff = monthly_obs(rd("fred_GS10")), monthly_obs(rd("fred_TB3MS")), monthly_obs(rd("fred_FEDFUNDS"))
    X["term"] = gs10 - tb3
    X["term_chg3"] = X["term"] - X["term"].shift(3)
    X["rate_chg12"] = gs10 - gs10.shift(12)
    X["ff_chg6"] = ff - ff.shift(6)
    X["credit"] = monthly_obs(rd("fred_BAA")) - monthly_obs(rd("fred_AAA"))
    X["credit_chg3"] = X["credit"] - X["credit"].shift(3)
    cpi = monthly_obs(rd("fred_CPIAUCSL"), 1)
    X["cpi_yoy"] = cpi / cpi.shift(12) - 1
    X["cpi_trend"] = X["cpi_yoy"] - X["cpi_yoy"].shift(3)
    un = monthly_obs(rd("fred_UNRATE"), 1)
    X["unemp_chg6"] = un - un.shift(6)
    ip = monthly_obs(rd("fred_INDPRO"), 1)
    X["ip_yoy"] = ip / ip.shift(12) - 1
    cl = month_end_last(rd("fred_ICSA").rolling(4).mean(), lag_days=7)
    X["claims_yoy"] = cl / cl.shift(12) - 1
    X["nfci"] = month_end_last(rd("fred_NFCI"), lag_days=7)
    L = french_lag
    X["mkt_24m"] = ((1 + mkt_m).rolling(24).apply(np.prod, raw=True) - 1).shift(L)
    X["mkt_vol6"] = (mkt_d.rolling(126).std().resample("ME").last().reindex(months) * np.sqrt(252)).shift(L)
    X["umd_vol6"] = (mom.rolling(126).std().resample("ME").last().reindex(months) * np.sqrt(252)).shift(L)
    X["umd_12m"] = ((1 + umd_m).rolling(12).apply(np.prod, raw=True) - 1).shift(L)
    # Lag 0 keeps the research's exact (unclipped) value at the data edge; with a lag,
    # carry the last known French reading forward like a live run would.
    if L > 0:
        for c in ("mkt_24m", "mkt_vol6", "umd_vol6", "umd_12m"):
            X[c] = X[c].ffill(limit=2)
    Z = (X - X.expanding(60).mean()) / X.expanding(60).std()

    nxt = umd_m.shift(-1)
    pred = pd.Series(np.nan, index=months)
    for i, t in enumerate(months):
        if t < pd.Timestamp("1977-01-31"):
            continue
        # rows s whose outcome month s+1 is known at t: s+1 <= t-L  ->  s < t-L
        trn = months[:max(i - L, 0)]
        d = pd.concat([Z.loc[trn], nxt.loc[trn]], axis=1).dropna()
        if len(d) < 60 or Z.loc[t].isna().any():
            continue
        A = np.column_stack([np.ones(len(d)), d.iloc[:, :-1].values])
        pen = RIDGE_ALPHA * np.eye(A.shape[1]); pen[0, 0] = 0
        beta = np.linalg.solve(A.T @ A + pen, A.T @ d.iloc[:, -1].values)
        pred.loc[t] = np.concatenate([[1.0], Z.loc[t].values]) @ beta
    tv = X["umd_vol6"].loc[TARGET_VOL_DEV[0]:TARGET_VOL_DEV[1]].median()
    volw = (tv / X["umd_vol6"]).clip(0, 2)
    on = (pred > 0).astype(float).where(pred.notna())
    weight = volw * on
    detail = pd.DataFrame({"weight": weight, "vol_scale": volw, "ridge_pred": pred, "ridge_on": on,
                           "umd_vol6": X["umd_vol6"], "next_umd": nxt})
    return weight, detail
