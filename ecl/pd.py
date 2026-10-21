"""
Point-in-time PD term structures.

The 1-year TTC transition matrix of each segment is conditioned on a credit
cycle index Z (one-factor Vasicek / CreditMetrics): for a row i, with states
ordered from default upwards, the cumulative probability of ending in a state
at or below j becomes

    C_ij(Z) = Phi( (Phi^-1(C_ij) - sqrt(rho) Z) / sqrt(1 - rho) ).

Because E_Z[C_ij(Z)] = C_ij for Z ~ N(0, 1), the TTC matrix is the average of
the PIT matrices over the cycle. Z is derived from the macroeconomic scenario
(GDP growth, unemployment) for the first three years and reverts to zero
afterwards: beyond the forecast horizon the yearly matrix is a mixture
phi^(t-3) M(Z_3) + (1 - phi^(t-3)) M_TTC, so that PDs revert to the long-run
(TTC) average rather than to the median Z = 0 matrix, which lies below TTC by
convexity. Lifetime PDs are obtained by chaining the yearly matrices.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from . import params as P


def z_forecast(macro: dict, loadings: dict | None = None, long_run: dict | None = None) -> np.ndarray:
    """Credit cycle index Z_t over the forecast years (positive = benign)."""
    loadings = loadings or P.Z_LOADINGS
    long_run = long_run or P.LONG_RUN
    n = len(next(iter(macro.values())))
    z = np.zeros(n)
    for var, (beta, sd) in loadings.items():
        z += beta * (np.asarray(macro[var], dtype=float) - long_run[var]) / sd
    return z


def house_price_index(macro: dict, horizon: int = P.MAX_HORIZON, shock: float = 0.0) -> np.ndarray:
    """House price index at the end of each year (index 0 = today = 1)."""
    g = list(np.asarray(macro.get("House prices (%)", []), dtype=float) / 100)
    g += [P.LONG_RUN["House prices (%)"] / 100] * (horizon - len(g))
    idx = np.concatenate([[1.0], np.cumprod(1 + np.asarray(g[:horizon]))])
    return idx * (1 + shock)


def conditional_matrix(m: np.ndarray, z: float, rho: float) -> np.ndarray:
    """PIT transition matrix conditional on Z (rows from non-default states)."""
    n = m.shape[0]
    out = np.zeros_like(m)
    for i in range(n - 1):
        row_rev = m[i, ::-1]                       # default first, best grade last
        cum = np.clip(np.cumsum(row_rev), 0.0, 1.0)
        c = np.empty_like(cum)
        c[-1] = 1.0
        inner = cum[:-1]
        with np.errstate(divide="ignore"):
            thr = norm.ppf(np.clip(inner, 1e-15, 1 - 1e-15))
        c[:-1] = np.where(inner <= 0, 0.0, np.where(inner >= 1, 1.0,
                          norm.cdf((thr - np.sqrt(rho) * z) / np.sqrt(1 - rho))))
        c = np.maximum.accumulate(c)
        out[i] = np.diff(np.concatenate([[0.0], c]))[::-1]
    out[-1, -1] = 1.0
    return out


def yearly_matrices(m: np.ndarray, z_fore: np.ndarray, rho: float, horizon: int = P.MAX_HORIZON,
                    phi: float = P.MEAN_REVERSION) -> list:
    """PIT matrices for years 1..horizon: conditional on Z over the forecast years, then mixed
    back towards the TTC matrix."""
    n_f = len(z_fore)
    mats = [conditional_matrix(m, z, rho) for z in z_fore[:horizon]]
    last = mats[-1] if mats else m
    for t in range(n_f, horizon):
        w = phi ** (t - n_f + 1)
        mats.append(w * last + (1 - w) * m)
    return mats


def cumulative_pd(m: np.ndarray, z_fore: np.ndarray, rho: float, horizon: int = P.MAX_HORIZON,
                  phi: float = P.MEAN_REVERSION) -> np.ndarray:
    """Cumulative default probability from each non-default grade at the end of years 1..H.
    Returns an array (n_grades, H)."""
    n = m.shape[0]
    state = np.eye(n)[: n - 1]
    out = np.zeros((n - 1, horizon))
    for t, mt in enumerate(yearly_matrices(m, z_fore, rho, horizon, phi)):
        state = state @ mt
        out[:, t] = state[:, -1]
    return out


def cumulative_pd_ttc(m: np.ndarray, horizon: int = P.MAX_HORIZON) -> np.ndarray:
    n = m.shape[0]
    state = np.eye(n)[: n - 1]
    out = np.zeros((n - 1, horizon))
    for t in range(horizon):
        state = state @ m
        out[:, t] = state[:, -1]
    return out


def pd_tables(matrices: dict, scenarios: dict, rho: dict, horizon: int = P.MAX_HORIZON) -> dict:
    """{segment: {scenario: cumulative PD (n_grades, H)}}, plus the 'TTC' entry."""
    out = {}
    for seg, m in matrices.items():
        out[seg] = {s: cumulative_pd(m, z_forecast(mac), rho[seg], horizon) for s, (_, mac) in scenarios.items()}
        out[seg]["TTC"] = cumulative_pd_ttc(m, horizon)
    return out


def lifetime_from_cum(cum: np.ndarray, grade_idx: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Cumulative PD over a (possibly fractional) number of years for each loan, by log-linear
    interpolation of survival between yearly points."""
    years = np.clip(years, 1e-6, cum.shape[1])
    lo = np.floor(years).astype(int)
    frac = years - lo
    s = 1.0 - cum[grade_idx]                       # survival (n_loans, H)
    s = np.concatenate([np.ones((len(grade_idx), 1)), s], axis=1)
    s_lo = s[np.arange(len(grade_idx)), lo]
    s_hi = s[np.arange(len(grade_idx)), np.minimum(lo + 1, cum.shape[1])]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(s_lo > 0, s_hi / s_lo, 0.0)
    return 1.0 - s_lo * np.power(np.clip(ratio, 0, 1), frac)
