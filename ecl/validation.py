"""
PD model validation on the one-year observation window (opening -> closing):
calibration tests by grade (one-sided binomial and Jeffreys tests, as in the
ECB instructions for reporting the validation results of internal models),
discriminatory power (AUC / Gini, Mann-Whitney) and population stability (PSI).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta, binom, rankdata

from . import demo
from . import engine as E
from . import params as P


def observation_sample(portfolio: pd.DataFrame, st: E.Settings, scen_open: dict, pd_basis: str = "PIT"
                       ) -> pd.DataFrame:
    o = demo.snapshot(portfolio, "open")
    r = E.compute_ecl(o, st, scen_open)
    perf = r["rating"].ne("D").to_numpy()
    status = portfolio.set_index("id").loc[r["id"], "status"].to_numpy()
    close_rating = portfolio.set_index("id").loc[r["id"], "rating"].to_numpy()
    s = pd.DataFrame({"id": r["id"], "segment": r["segment"], "grade": r["rating"], "pd_pit": r["pd12"],
                      "default": (status == "Existing") & (close_rating == "D")})
    gidx, _ = E.grade_index(r["segment"], r["rating"])
    s["pd_ttc"] = [st.matrices[seg][g, -1] for seg, g in zip(r["segment"], gidx)]
    s["pd"] = s["pd_pit"] if pd_basis == "PIT" else s["pd_ttc"]
    return s[perf].reset_index(drop=True)


MIN_N = 30          # grades with fewer obligors are not tested


def calibration(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seg, g), d in sample.groupby(["segment", "grade"]):
        n, k, pdm = len(d), int(d["default"].sum()), float(d["pd"].mean())
        p_bin = float(binom.sf(k - 1, n, pdm)) if k > 0 else 1.0       # P(X >= k | PD)
        p_jef = float(beta.cdf(pdm, k + 0.5, n - k + 0.5))              # P(DR <= PD | data)
        p_min = min(p_bin, p_jef)
        flag = ("Too few observations" if n < MIN_N else
                "Red" if p_min < 0.01 else "Amber" if p_min < 0.05 else "Green")
        rows.append({"segment": seg, "grade": g, "N": n, "defaults": k, "PD (avg)": pdm, "ODR": k / n,
                     "binomial p": p_bin, "Jeffreys p": p_jef, "result": flag})
    out = pd.DataFrame(rows)
    order = {g: i for sc in P.SCALES.values() for i, g in enumerate(sc)}
    return out.sort_values(["segment", "grade"], key=lambda c: c.map(order) if c.name == "grade" else c)


def auc(score: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    rk = rankdata(score)
    return float((rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def discrimination(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seg, d in sample.groupby("segment"):
        a = auc(d["pd"].to_numpy(), d["default"].to_numpy())
        n1 = int(d["default"].sum())
        n0 = len(d) - n1
        # Hanley-McNeil standard error
        q1, q2 = a / (2 - a), 2 * a ** 2 / (1 + a)
        se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a ** 2) + (n0 - 1) * (q2 - a ** 2)) / max(n1 * n0, 1))
        rows.append({"segment": seg, "N": len(d), "defaults": n1, "AUC": a, "AUC 95% CI low": max(a - 1.96 * se, 0.0),
                     "AUC 95% CI high": min(a + 1.96 * se, 1.0), "Gini": 2 * a - 1})
    return pd.DataFrame(rows)


def psi(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Grade distribution of performing exposures at opening (reference) vs closing."""
    rows = []
    for seg, d in portfolio.groupby("segment"):
        scale = P.SCALES[P.SEGMENT_SCALE[seg]][:-1]
        ref = d.loc[d["rating_open"].isin(scale), "rating_open"].value_counts().reindex(scale, fill_value=0)
        cur = d.loc[d["status"].isin(["Existing", "New"]) & d["rating"].isin(scale), "rating"] \
            .value_counts().reindex(scale, fill_value=0)
        e = (ref / ref.sum()).clip(lower=1e-4)
        a = (cur / cur.sum()).clip(lower=1e-4)
        contrib = (a - e) * np.log(a / e)
        for g in scale:
            rows.append({"segment": seg, "grade": g, "reference %": e[g], "current %": a[g], "PSI contribution": contrib[g]})
    out = pd.DataFrame(rows)
    tot = out.groupby("segment")["PSI contribution"].sum().rename("PSI").reset_index()
    tot["result"] = np.where(tot["PSI"] < 0.10, "Stable", np.where(tot["PSI"] < 0.25, "Moderate shift", "Significant shift"))
    return out, tot
