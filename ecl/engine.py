"""
IFRS 9 expected credit loss engine.

For each exposure and each macroeconomic scenario s:

    ECL_s = sum_t  mPD_{s,t} x EAD_t x LGD_{s,t} x DF_t

with yearly marginal PDs from the PIT term structure of the current grade,
EAD at mid-year (amortisation, prepayments, behavioural CCF on undrawn limits),
scenario-dependent LGD (collateral indexed on house prices for mortgages,
cyclical unsecured LGD otherwise) and discounting at the effective interest
rate (IFRS 9 B5.5.44). Stage 1 uses the first year only (12-month ECL),
Stage 2 the whole remaining life, Stage 3 PD = 100%. The reported ECL is the
probability-weighted average over scenarios (IFRS 9 5.5.17).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import params as P
from . import pd as PDM

H = P.MAX_HORIZON

# Columns of a reporting-date snapshot
SNAP_COLUMNS = ["id", "segment", "product", "amort", "balance", "limit", "rate", "remaining", "rating_orig",
                "rating", "dpd", "utp", "watchlist", "forborne", "restructured", "cure_from", "months_since_cure",
                "collateral", "collateral_type", "turnover"]


@dataclass
class Settings:
    matrices: dict = field(default_factory=lambda: {s: P.build_ttc_matrix(s) for s in P.TTC_PD})
    rho: dict = field(default_factory=lambda: dict(P.RHO))
    sicr: dict = field(default_factory=lambda: dict(P.SICR_DEFAULTS))
    products: dict = field(default_factory=lambda: {k: dict(v) for k, v in P.PRODUCTS.items()})
    lgd: dict = field(default_factory=lambda: {k: dict(v) for k, v in P.LGD.items()})


# ==========================================================================
# Helpers
# ==========================================================================


def grade_index(segment: pd.Series, rating: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Index of the grade in its scale and default flag."""
    seg = pd.Series(np.asarray(segment)).map(P.SEGMENT_SCALE).to_numpy()
    rat = np.asarray(rating, dtype=object)
    j = np.full(len(rat), -1)
    for name, scale in P.SCALES.items():
        lookup = {g: k for k, g in enumerate(scale)}
        msk = seg == name
        j[msk] = pd.Series(rat[msk]).map(lookup).fillna(len(scale) - 1).astype(int).to_numpy()
    n_states = np.array([len(P.SCALES[x]) if isinstance(x, str) else 8 for x in seg])
    dflt = j == n_states - 1
    return np.minimum(j, n_states - 2), dflt


def _cum_at(cum: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Cumulative PD of each row of `cum` (n, H) at fractional times `years` (n, k)."""
    n = cum.shape[0]
    s = np.concatenate([np.ones((n, 1)), 1.0 - cum], axis=1)
    y = np.clip(years, 0.0, cum.shape[1])
    lo = np.floor(y).astype(int)
    hi = np.minimum(lo + 1, cum.shape[1])
    frac = y - lo
    rows = np.arange(n)[:, None]
    s_lo, s_hi = s[rows, lo], s[rows, hi]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(s_lo > 0, s_hi / s_lo, 0.0)
    return 1.0 - s_lo * np.power(np.clip(ratio, 0.0, 1.0), frac)


def _z_full(macro: dict) -> np.ndarray:
    z = PDM.z_forecast(macro)
    out = np.zeros(H)
    out[: len(z)] = z[:H]
    for t in range(len(z), H):
        out[t] = z[-1] * P.MEAN_REVERSION ** (t - len(z) + 1)
    return out


def _horizon(df: pd.DataFrame, st: Settings) -> np.ndarray:
    rem = df["remaining"].to_numpy(dtype=float)
    life = np.array([st.products[p]["life"] or 0 for p in df["product"]], dtype=float)
    rev = df["amort"].to_numpy() == "Revolving"
    rem = np.where(rev, np.where(life > 0, life, np.maximum(rem, 1.0)), rem)
    return np.clip(rem, 1 / 12, H)


def ead_profile(df: pd.DataFrame, st: Settings, tau: np.ndarray) -> np.ndarray:
    """Expected exposure at times tau (n, k), conditional on no prior default."""
    b = df["balance"].to_numpy(dtype=float)[:, None]
    lim = df["limit"].to_numpy(dtype=float)[:, None]
    r = df["rate"].to_numpy(dtype=float)[:, None] / 100
    rem = df["remaining"].to_numpy(dtype=float)[:, None].clip(1 / 12, None)
    amort = df["amort"].to_numpy()[:, None]
    cpr = np.array([st.products[p]["cpr"] for p in df["product"]])[:, None]
    ccf = np.array([st.products[p]["ccf"] for p in df["product"]])[:, None]
    surv_prep = (1 - cpr) ** tau
    i = r / 12
    n_m = rem * 12
    with np.errstate(divide="ignore", invalid="ignore"):
        ann = np.where(i > 0, 1 - ((1 + i) ** (12 * tau) - 1) / ((1 + i) ** n_m - 1), 1 - tau / rem)
    ann = np.clip(ann, 0, 1)
    lin = np.clip(1 - tau / rem, 0, 1)
    out = np.where(amort == "Annuity", b * ann * surv_prep,
          np.where(amort == "Linear", b * lin * surv_prep,
          np.where(amort == "Bullet", b * (tau < rem) * surv_prep,
                   b + ccf * np.maximum(lim - b, 0))))
    return out


def lgd_profile(df: pd.DataFrame, st: Settings, ead: np.ndarray, tau: np.ndarray, z: np.ndarray,
                hpi: np.ndarray) -> np.ndarray:
    """Scenario LGD at default times tau. z: credit cycle per year (H,), hpi: index at year ends (H+1,)."""
    seg = df["segment"].to_numpy()[:, None]
    r = df["rate"].to_numpy(dtype=float)[:, None] / 100
    col = df["collateral"].fillna(0).to_numpy(dtype=float)[:, None]
    ctype = df["collateral_type"].fillna("").to_numpy()[:, None]
    yr = np.clip(np.ceil(tau).astype(int) - 1, 0, H - 1)
    zt = z[yr]
    segs = pd.Series(df["segment"].to_numpy())
    par = {k: segs.map({s_: v[k] for s_, v in st.lgd.items()}).to_numpy(dtype=float)[:, None]
           for k in ("unsecured", "cure", "haircut", "sensitivity", "cost")}
    lgd_u = np.clip(par["unsecured"] - par["sensitivity"] * zt, 0, 1)
    # collateral value at the time of sale (house price index for property)
    t_sale = np.clip(tau + P.TIME_TO_SALE, 0, H)
    hpi_sale = np.interp(t_sale, np.arange(H + 1), hpi)
    prop = (ctype == "RRE") | (ctype == "CRE")
    c_val = np.where(prop, col * hpi_sale, col)
    rec = c_val * (1 - par["haircut"]) / (1 + r) ** P.TIME_TO_SALE
    with np.errstate(divide="ignore", invalid="ignore"):
        shortfall = np.where(ead > 0, np.clip(ead - rec, 0, None) / ead, 0.0)
    lgd = (1 - par["cure"]) * (shortfall * lgd_u + par["cost"])
    return np.clip(lgd, 0, 1)


# ==========================================================================
# Core calculation
# ==========================================================================


@dataclass
class ScenarioProfile:
    tau: np.ndarray
    mpd: np.ndarray
    ead: np.ndarray
    lgd: np.ndarray
    df: np.ndarray
    pd12: np.ndarray
    pd_life: np.ndarray
    ecl12: np.ndarray
    ecl_life: np.ndarray
    ecl_default: np.ndarray
    lgd_default: np.ndarray


def profiles(df: pd.DataFrame, st: Settings, scenarios: dict) -> dict:
    """Yearly PD / EAD / LGD / DF profiles and ECL by scenario."""
    tables = PDM.pd_tables(st.matrices, scenarios, st.rho)
    gidx, _ = grade_index(df["segment"], df["rating"])
    rem = _horizon(df, st)
    t = np.arange(1, H + 1)[None, :]
    start = (t - 1).astype(float)
    end = np.minimum(t, rem[:, None])
    active = start < rem[:, None]
    tau = np.where(active, (start + end) / 2, 0.0)
    r = df["rate"].to_numpy(dtype=float)[:, None] / 100
    disc = np.where(active, (1 + r) ** -tau, 0.0)
    ead = np.where(active, ead_profile(df, st, tau), 0.0)
    ead_now = ead_profile(df, st, np.zeros((len(df), 1)))[:, 0]
    out = {}
    segs = df["segment"].to_numpy()
    for s, (_, mac) in scenarios.items():
        cum = np.zeros((len(df), H))
        for seg in np.unique(segs):
            msk = segs == seg
            cum[msk] = tables[seg][s][gidx[msk]]
        c_end = _cum_at(cum, end)
        c_start = _cum_at(cum, np.minimum(start, rem[:, None]))
        mpd = np.where(active, np.clip(c_end - c_start, 0, 1), 0.0)
        z = _z_full(mac)
        hpi = PDM.house_price_index(mac)
        lgd = np.where(active, lgd_profile(df, st, ead, tau, z, hpi), 0.0)
        contrib = mpd * ead * lgd * disc
        lgd0 = lgd_profile(df, st, ead_now[:, None], np.zeros((len(df), 1)), z, hpi)[:, 0]
        out[s] = ScenarioProfile(tau=tau, mpd=mpd, ead=ead, lgd=lgd, df=disc,
                                 pd12=mpd[:, 0], pd_life=c_end[np.arange(len(df)), np.ceil(rem).astype(int) - 1],
                                 ecl12=contrib[:, 0], ecl_life=contrib.sum(axis=1),
                                 ecl_default=ead_now * lgd0, lgd_default=lgd0)
    return out


def ttc_lifetime_pd(df: pd.DataFrame, st: Settings, rating_col: str) -> np.ndarray:
    gidx, dflt = grade_index(df["segment"], df[rating_col])
    rem = _horizon(df, st)
    out = np.zeros(len(df))
    for seg in df["segment"].unique():
        msk = (df["segment"] == seg).to_numpy()
        cum = PDM.cumulative_pd_ttc(st.matrices[seg])[gidx[msk]]
        out[msk] = _cum_at(cum, rem[msk][:, None])[:, 0]
    return np.where(dflt, 1.0, out)


def assign_stage(df: pd.DataFrame, pd_now: np.ndarray, pd_orig: np.ndarray, sicr: dict) -> tuple:
    """Stage and the first trigger met."""
    _, dflt = grade_index(df["segment"], df["rating"])
    dpd = df["dpd"].to_numpy(dtype=float)
    utp = df["utp"].astype(bool).to_numpy()
    cure = df["cure_from"].fillna("").to_numpy()
    months = df["months_since_cure"].fillna(99).to_numpy(dtype=float)
    restr = df["restructured"].astype(bool).to_numpy()
    prob_s3 = np.where(restr, sicr["probation_s3_restructured_months"], sicr["probation_s3_months"])
    n = len(df)
    stage = np.ones(n, dtype=int)
    reason = np.array(["Performing"] * n, dtype=object)

    s3_rules = [(dflt, "Default grade"), (dpd >= 90, "90 days past due"), (utp, "Unlikely to pay"),
                ((cure == "S3") & (months < prob_s3), "Default probation")]
    lcr = np.array([sicr["low_credit_risk"] and seg == "Corporate" and r in P.INVESTMENT_GRADE
                    for seg, r in zip(df["segment"], df["rating"])])
    rel = sicr["relative"]
    rel = np.array([rel[s] if isinstance(rel, dict) else rel for s in df["segment"]], dtype=float)
    quant = (pd_now >= rel * pd_orig) & (pd_now - pd_orig >= sicr["absolute_bp"] / 1e4) & ~lcr
    s2_rules = [(quant, "PD deterioration"),
                ((dpd >= 30) & sicr["backstop_30dpd"], "30 days past due"),
                (df["watchlist"].astype(bool).to_numpy() & sicr["watchlist"], "Watchlist"),
                (df["forborne"].astype(bool).to_numpy() & sicr["forborne"], "Forborne"),
                ((cure == "S2") & (months < sicr["probation_s2_months"]), "Stage 2 probation"),
                ((cure == "S3") & (months >= prob_s3) & (months < prob_s3 + sicr["probation_s2_months"]),
                 "Post-default probation")]
    for cond, why in reversed(s2_rules):
        stage = np.where(cond, 2, stage)
        reason = np.where(cond, why, reason)
    lcr_kept = lcr & (stage == 1) & (pd_now >= rel * pd_orig) & \
        (pd_now - pd_orig >= sicr["absolute_bp"] / 1e4)
    reason = np.where(lcr_kept, "Low credit risk exemption", reason)
    for cond, why in reversed(s3_rules):
        stage = np.where(cond, 3, stage)
        reason = np.where(cond, why, reason)
    return stage, reason


def compute_ecl(df: pd.DataFrame, st: Settings, scenarios: dict, force_stage: np.ndarray | None = None
                ) -> pd.DataFrame:
    """Per-exposure results (model ECL before post-model adjustments)."""
    prof = profiles(df, st, scenarios)
    w = {s: v[0] for s, v in scenarios.items()}
    tot_w = sum(w.values()) or 1.0
    w = {s: v / tot_w for s, v in w.items()}
    wsum = lambda attr: sum(w[s] * getattr(prof[s], attr) for s in prof)  # noqa: E731
    pd12, pd_life = wsum("pd12"), wsum("pd_life")
    pd_orig = ttc_lifetime_pd(df, st, "rating_orig")
    stage, reason = assign_stage(df, pd_life, pd_orig, st.sicr)
    if force_stage is not None:
        stage = np.asarray(force_stage)
    ead_now = ead_profile(df, st, np.zeros((len(df), 1)))[:, 0]
    res = pd.DataFrame({
        "id": df["id"].to_numpy(), "segment": df["segment"].to_numpy(), "product": df["product"].to_numpy(),
        "rating": df["rating"].to_numpy(), "stage": stage, "reason": reason,
        "gross": df["balance"].to_numpy(dtype=float), "ead": ead_now,
        "pd12": pd12, "pd_life": pd_life, "pd_orig_life": pd_orig,
        "ecl12": wsum("ecl12"), "ecl_life": wsum("ecl_life"), "ecl_default": wsum("ecl_default"),
        "lgd_default": wsum("lgd_default"),
    })
    pick = lambda e12, el, ed: np.where(stage == 1, e12, np.where(stage == 2, el, ed))  # noqa: E731
    for s, p in prof.items():
        res[f"ECL | {s}"] = pick(p.ecl12, p.ecl_life, p.ecl_default)
    res["ecl"] = pick(res["ecl12"], res["ecl_life"], res["ecl_default"])
    res["ecl_all_lifetime"] = np.where(stage == 3, res["ecl_default"], res["ecl_life"])
    return res


def explain(row: pd.DataFrame, st: Settings, scenarios: dict) -> pd.DataFrame:
    """Year-by-year ECL build-up for one exposure, probability-weighted across scenarios."""
    prof = profiles(row, st, scenarios)
    w = {s: v[0] for s, v in scenarios.items()}
    tw = sum(w.values()) or 1.0
    n = int(np.ceil(_horizon(row, st)[0]))
    rows = []
    for t in range(n):
        d = {"year": t + 1, "time of default (y)": prof[next(iter(prof))].tau[0, t]}
        for s, p in prof.items():
            d[f"mPD {s}"] = p.mpd[0, t]
        d["mPD weighted"] = sum(w[s] / tw * p.mpd[0, t] for s, p in prof.items())
        d["EAD"] = prof[next(iter(prof))].ead[0, t]
        d["LGD weighted"] = sum(w[s] / tw * p.lgd[0, t] for s, p in prof.items())
        d["DF"] = prof[next(iter(prof))].df[0, t]
        d["ECL contribution"] = sum(w[s] / tw * p.mpd[0, t] * p.ead[0, t] * p.lgd[0, t] * p.df[0, t]
                                    for s, p in prof.items())
        rows.append(d)
    return pd.DataFrame(rows)


# ==========================================================================
# Post-model adjustments and summaries
# ==========================================================================

OVERLAY_COLUMNS = ["segment", "stage", "type", "value", "rationale"]


def default_overlays() -> pd.DataFrame:
    return pd.DataFrame([
        ("Mortgage", "All", "% of ECL", 10.0, "Transition risk: energy-inefficient collateral (novel risk)"),
        ("Corporate", "Stage 2", "Amount", 4.0, "Sector overlay: commercial real estate refinancing risk"),
        ("Consumer", "All", "% of ECL", 5.0, "Inflation pressure on household affordability"),
    ], columns=OVERLAY_COLUMNS)


def apply_overlays(res: pd.DataFrame, overlays: pd.DataFrame) -> pd.DataFrame:
    """Allocate post-model adjustments to exposures pro rata to modelled ECL within the scope."""
    out = res.copy()
    out["pma"] = 0.0
    for _, o in overlays.dropna(subset=["segment", "value"]).iterrows():
        scope = out["segment"].eq(o["segment"]) | (o["segment"] == "All")
        if str(o["stage"]).startswith("Stage"):
            scope &= out["stage"].eq(int(str(o["stage"])[-1]))
        base = out.loc[scope, "ecl"]
        if base.sum() <= 0:
            continue
        add = base * float(o["value"]) / 100 if o["type"] == "% of ECL" else base / base.sum() * float(o["value"])
        out.loc[scope, "pma"] += add
    out["ecl_total"] = out["ecl"] + out["pma"]
    return out


def stage_summary(res: pd.DataFrame, by: str = "segment") -> pd.DataFrame:
    g = res.groupby([by, "stage"]).agg(gross=("gross", "sum"), ecl=("ecl_total", "sum"), n=("id", "count"))
    tbl = g.unstack("stage").fillna(0)
    out = pd.DataFrame(index=tbl.index)
    for s in (1, 2, 3):
        gs = tbl[("gross", s)] if ("gross", s) in tbl else 0
        es = tbl[("ecl", s)] if ("ecl", s) in tbl else 0
        out[f"Stage {s} gross"] = gs
        out[f"Stage {s} ECL"] = es
        out[f"Stage {s} coverage"] = np.where(np.asarray(gs) > 0, np.asarray(es) / np.where(np.asarray(gs) > 0, gs, 1), 0)
    out.loc["Total"] = out.sum()
    for s in (1, 2, 3):
        gs = out.loc["Total", f"Stage {s} gross"]
        out.loc["Total", f"Stage {s} coverage"] = out.loc["Total", f"Stage {s} ECL"] / gs if gs else 0
    return out
