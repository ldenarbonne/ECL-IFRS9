"""
Synthetic loan book observed at two dates (opening = one year ago, closing =
reporting date), so that the IFRS 7 reconciliation and the PD backtest can be
run. Ratings migrate over the year with the PIT matrix conditional on the
realised credit cycle (params.REALISED_Z). All data are fictitious.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import params as P
from . import pd as PDM

PORTFOLIO_COLUMNS = [
    "id", "segment", "product", "amort", "status", "orig_term", "age_open", "age", "limit", "balance_open",
    "balance", "rate", "rating_orig", "rating_open", "rating", "dpd_open", "dpd", "utp_open", "utp",
    "watchlist_open", "watchlist", "forborne_open", "forborne", "restructured", "cure_from", "months_since_cure",
    "collateral_open", "collateral", "collateral_type", "turnover",
]

SIZES = {"Corporate term loan": 320, "Corporate RCF": 80, "Mortgage": 5000, "Consumer loan": 2500,
         "Credit card": 2500}
ORIG_DIST = {"Corporate": [0.02, 0.06, 0.17, 0.35, 0.25, 0.13, 0.02],
             "Mortgage": [0.25, 0.25, 0.20, 0.15, 0.10, 0.04, 0.01],
             "Consumer": [0.15, 0.20, 0.22, 0.20, 0.13, 0.07, 0.03]}
DPD30_PROB = {"Corporate": [0, 0, 0, 0, 0.005, 0.02, 0.10],
              "Mortgage": [0, 0, 0.002, 0.01, 0.04, 0.12, 0.30],
              "Consumer": [0, 0.002, 0.005, 0.015, 0.05, 0.15, 0.35]}


def _sample_rows(m: np.ndarray, start: np.ndarray, rng) -> np.ndarray:
    cum = np.cumsum(m[start], axis=1)
    u = rng.random(len(start))[:, None]
    return np.minimum((u > cum).sum(axis=1), m.shape[0] - 1)


def _balance(b0, rate, term, age, amort):
    i = rate / 1200
    n = term * 12
    k = np.clip(age, 0, term) * 12
    with np.errstate(divide="ignore", invalid="ignore"):
        ann = b0 * ((1 + i) ** n - (1 + i) ** k) / ((1 + i) ** n - 1)
    lin = b0 * (1 - k / n)
    bul = np.where(age < term, b0, 0.0)
    return np.where(amort == "Annuity", ann, np.where(amort == "Linear", lin, bul))


def _dpd(seg, grade, scale_len, rng):
    n = len(grade)
    out = np.zeros(n)
    p30 = np.array([DPD30_PROB[s][min(g, 6)] if g < scale_len - 1 else 0 for s, g in zip(seg, grade)])
    u = rng.random(n)
    out = np.where(u < p30, rng.integers(30, 90, n), np.where(u < p30 * 3, rng.integers(1, 30, n), 0))
    return out


def generate(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mats = {s: P.build_ttc_matrix(s) for s in P.TTC_PD}
    pit = {s: PDM.conditional_matrix(mats[s], P.REALISED_Z, P.RHO[s]) for s in mats}
    frames = []
    for prod, n_open in SIZES.items():
        cfg = P.PRODUCTS[prod]
        seg = cfg["segment"]
        n_new = int(n_open * 0.12)
        n = n_open + n_new
        new = np.r_[np.zeros(n_open, bool), np.ones(n_new, bool)]
        if seg == "Corporate":
            term = rng.choice([3, 5, 7], n).astype(float)
            b0 = np.exp(rng.normal(np.log(5), 0.6, n))
            rate = rng.uniform(3.5, 6.0, n)
            turnover = np.exp(rng.normal(np.log(120), 1.0, n))
        elif seg == "Mortgage":
            term = rng.choice([20, 25], n).astype(float)
            b0 = rng.uniform(0.15, 0.60, n)
            rate = rng.uniform(1.5, 4.5, n)
            turnover = np.full(n, np.nan)
        else:
            term = rng.choice([3, 5, 7], n).astype(float) if prod == "Consumer loan" else np.full(n, 12.0)
            b0 = rng.uniform(0.01, 0.06, n) if prod == "Consumer loan" else rng.uniform(0.003, 0.02, n)
            rate = rng.uniform(5, 10, n) if prod == "Consumer loan" else rng.uniform(15, 19, n)
            turnover = np.full(n, np.nan)
        amort = np.full(n, cfg["amort"], dtype=object)
        if prod == "Corporate term loan":
            amort = np.where(rng.random(n) < 0.3, "Bullet", "Linear")
        age_max = np.minimum(term, 12)
        age_open = np.where(new, np.nan, rng.uniform(0, 1, n) * age_max)
        age_close = np.where(new, rng.uniform(0, 1, n), age_open + 1)
        orig = rng.choice(7, n, p=ORIG_DIST[seg])
        # rating one year ago: TTC migration from origination over the elapsed years
        g_open = orig.copy()
        yrs = np.where(new, 0, np.floor(np.nan_to_num(age_open)).astype(int))
        for y in range(int(yrs.max())):
            mv = yrs > y
            g_open[mv] = _sample_rows(mats[seg], g_open[mv], rng)
        # most exposures that defaulted before last year have been resolved or written off already
        old_d = (g_open == 7) & (rng.random(n) < 0.6)
        g_open[old_d] = orig[old_d]
        g_close = g_open.copy()
        perf = (~new) & (g_open < 7)
        g_close[perf] = _sample_rows(pit[seg], g_open[perf], rng)
        g_close[new] = orig[new]
        # outcome of the year
        rem_open = term - np.nan_to_num(age_open)
        status = np.where(new, "New", "Existing").astype(object)
        matured = (~new) & (rem_open <= 1) & (amort != "Revolving")
        prepaid = (~new) & (rng.random(n) < cfg["cpr"]) & (g_open < 7)
        status[matured | prepaid] = "Derecognised"
        d_open = (~new) & (g_open == 7)
        u = rng.random(n)
        status[d_open & (u < 0.40)] = "Written-off"
        cured = d_open & (u >= 0.40) & (u < 0.55)
        g_close[cured] = 5
        # balances
        if cfg["amort"] == "Revolving":
            limit = b0 * (3 if seg == "Corporate" else 1.0)
            draw_o = rng.beta(2, 3, n)
            bal_open = np.where(new, np.nan, limit * draw_o)
            draw_c = np.clip(draw_o * np.exp(rng.normal(0, 0.25, n)), 0.02, 1.0)
            draw_c = np.where(g_close == 7, np.maximum(draw_c, 0.9), draw_c)
            bal_close = limit * draw_c
        else:
            bal_open = np.where(new, np.nan, _balance(b0, rate, term, np.nan_to_num(age_open), amort))
            bal_close = _balance(b0, rate, term, age_close, amort)
            limit = bal_close.copy()
        # collateral
        ctype = np.full(n, "", dtype=object)
        coll_o = np.full(n, np.nan)
        coll_c = np.full(n, np.nan)
        if seg == "Mortgage":
            ctype[:] = "RRE"
            v0 = b0 / rng.uniform(0.60, 0.95, n)
            coll_o = np.where(new, np.nan, v0 * 1.03 ** np.nan_to_num(age_open))
            coll_c = np.where(new, v0, coll_o * 1.02)
        elif seg == "Corporate":
            sec = rng.random(n) < 0.4
            ctype[sec] = np.where(rng.random(sec.sum()) < 0.6, "CRE", "Other")
            v = np.where(sec, b0 * rng.uniform(0.5, 1.2, n), np.nan)
            coll_o, coll_c = np.where(new, np.nan, v), v
        # delinquency and qualitative flags
        dpd_o = np.where(new, 0, _dpd([seg] * n, g_open, 8, rng))
        dpd_c = _dpd([seg] * n, g_close, 8, rng)
        dfl_o, dfl_c = g_open == 7, g_close == 7
        dpd_o = np.where(dfl_o, np.where(rng.random(n) < 0.75, rng.integers(90, 400, n), rng.integers(0, 60, n)), dpd_o)
        dpd_c = np.where(dfl_c, np.where(rng.random(n) < 0.75, rng.integers(90, 400, n), rng.integers(0, 60, n)), dpd_c)
        utp_o, utp_c = dfl_o & (dpd_o < 90), dfl_c & (dpd_c < 90)
        bad = 5 if seg == "Corporate" else 6
        watch_c = (seg == "Corporate") & (((g_close >= bad) & (rng.random(n) < 0.25)) |
                                           ((g_close - orig >= 3) & (rng.random(n) < 0.3)))
        watch_o = (seg == "Corporate") & (g_open >= bad) & (rng.random(n) < 0.25)
        forb_c = (g_close >= 5) & (g_close < 7) & (rng.random(n) < 0.10)
        forb_o = (g_open >= 5) & (g_open < 7) & (rng.random(n) < 0.10)
        restructured = cured & (rng.random(n) < 0.5)
        forb_c |= restructured
        cure_from = np.full(n, "", dtype=object)
        months = np.full(n, np.nan)
        cure_from[cured] = "S3"
        s2c = (~new) & (dpd_o >= 30) & (dpd_c < 30) & ~dfl_c & ~cured
        cure_from[s2c] = "S2"
        cm = cured | s2c
        months[cm] = rng.integers(0, 12, cm.sum())
        scale = P.SCALES[P.SEGMENT_SCALE[seg]]
        frames.append(pd.DataFrame({
            "segment": seg, "product": prod, "amort": amort, "status": status, "orig_term": term,
            "age_open": np.round(age_open, 3), "age": np.round(age_close, 3), "limit": limit,
            "balance_open": bal_open, "balance": bal_close, "rate": np.round(rate, 2),
            "rating_orig": [scale[i] for i in orig], "rating_open": [scale[i] if not nw else "" for i, nw in zip(g_open, new)],
            "rating": [scale[i] for i in g_close], "dpd_open": dpd_o, "dpd": dpd_c, "utp_open": utp_o, "utp": utp_c,
            "watchlist_open": watch_o, "watchlist": watch_c, "forborne_open": forb_o, "forborne": forb_c,
            "restructured": restructured, "cure_from": cure_from, "months_since_cure": months,
            "collateral_open": coll_o, "collateral": coll_c, "collateral_type": ctype, "turnover": turnover,
        }))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "id", [f"L{i:05d}" for i in range(len(df))])
    # closing values are irrelevant for exposures that left the book during the year
    gone = df["status"].isin(["Derecognised", "Written-off"])
    df.loc[gone, ["balance", "age", "dpd"]] = [0.0, np.nan, 0]
    for c in ["balance_open", "balance", "limit", "collateral_open", "collateral"]:
        df[c] = df[c].round(5)
    return df[PORTFOLIO_COLUMNS]


def snapshot(portfolio: pd.DataFrame, when: str = "close") -> pd.DataFrame:
    """Reporting-date view used by the engine (columns engine.SNAP_COLUMNS)."""
    from .engine import SNAP_COLUMNS
    if when == "close":
        d = portfolio[portfolio["status"].isin(["Existing", "New"])].copy()
        d["remaining"] = d["orig_term"] - d["age"]
    else:
        d = portfolio[portfolio["status"].isin(["Existing", "Derecognised", "Written-off"])].copy()
        d["remaining"] = d["orig_term"] - d["age_open"]
        d["balance"], d["rating"], d["dpd"], d["utp"] = d["balance_open"], d["rating_open"], d["dpd_open"], d["utp_open"]
        d["watchlist"], d["forborne"], d["collateral"] = d["watchlist_open"], d["forborne_open"], d["collateral_open"]
        d["restructured"] = False
        d["cure_from"], d["months_since_cure"] = "", np.nan
    d["remaining"] = d["remaining"].clip(lower=1 / 12)
    return d[SNAP_COLUMNS].reset_index(drop=True)
