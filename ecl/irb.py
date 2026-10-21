"""
IRB expected loss and capital for the same book, and comparison with IFRS 9
provisions (Basel Framework CRE31, CRE32, CRE35; CRR Articles 153-154, 158-159).

* PD: 1-year TTC PD of the current grade, floored at 0.05% (0.10% for QRRE
  revolvers).
* LGD: downturn LGD from the same LGD model under downturn conditions (Z = -2,
  house prices -25%), floored at the A-IRB input floors; or F-IRB supervisory
  values for corporates.
* Defaulted exposures: EL = ELBE x EAD (ELBE = best-estimate LGD used for Stage 3),
  K = max(0, LGD in default - ELBE).
* Risk weight functions without the 1.06 scaling factor (removed by the Basel III
  final reform).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import engine as E
from . import params as P


def irb_class(df: pd.DataFrame) -> np.ndarray:
    return np.select([df["segment"].eq("Corporate"), df["product"].eq("Mortgage"), df["product"].eq("Credit card")],
                     ["Corporate", "Residential mortgage", "QRRE"], "Other retail")


def risk_weight_k(cls, pd_, lgd, m, turnover):
    """Capital requirement K per unit of EAD (CRE31.5-31.15)."""
    pd_ = np.clip(pd_, 1e-6, 0.9999)
    g = norm.ppf(pd_)
    w50 = (1 - np.exp(-50 * pd_)) / (1 - np.exp(-50))
    w35 = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    s = np.clip(np.nan_to_num(turnover, nan=1e9), 5, 50)
    sme = np.where(np.nan_to_num(turnover, nan=1e9) <= 50, 0.04 * (1 - (s - 5) / 45), 0.0)
    r = np.select([cls == "Corporate", cls == "Residential mortgage", cls == "QRRE"],
                  [0.12 * w50 + 0.24 * (1 - w50) - sme, np.full_like(pd_, 0.15), np.full_like(pd_, 0.04)],
                  0.03 * w35 + 0.16 * (1 - w35))
    k = lgd * norm.cdf((g + np.sqrt(r) * norm.ppf(0.999)) / np.sqrt(1 - r)) - pd_ * lgd
    b = (0.11852 - 0.05478 * np.log(pd_)) ** 2
    ma = np.where(cls == "Corporate", (1 + (m - 2.5) * b) / (1 - 1.5 * b), 1.0)
    return np.maximum(k * ma, 0.0), r


def compute(snap: pd.DataFrame, res: pd.DataFrame, st: E.Settings, corporate_approach: str = "A-IRB"
            ) -> pd.DataFrame:
    cls = irb_class(snap)
    gidx, dflt = E.grade_index(snap["segment"], snap["rating"])
    pd_ttc = np.array([st.matrices[s][g, -1] for s, g in zip(snap["segment"], gidx)])
    floor = np.where(cls == "QRRE", P.IRB_PD_FLOOR["QRRE revolver"], P.IRB_PD_FLOOR["Corporate"])
    pd_irb = np.where(dflt, 1.0, np.maximum(pd_ttc, floor))
    bal = snap["balance"].to_numpy(dtype=float)
    undrawn = np.maximum(snap["limit"].to_numpy(dtype=float) - bal, 0)
    ccf_b = np.array([st.products[p]["ccf"] for p in snap["product"]])
    sa_ccf = np.where(cls == "Corporate", P.FIRB_CCF_COMMITMENT, 0.10)
    firb = (cls == "Corporate") & (corporate_approach == "F-IRB")
    ead = np.where(firb, bal + sa_ccf * undrawn, np.maximum(bal + ccf_b * undrawn, bal + 0.5 * sa_ccf * undrawn))
    # downturn LGD from the LGD model
    z_dt = np.full(P.MAX_HORIZON, P.DOWNTURN["z"])
    hpi_dt = np.full(P.MAX_HORIZON + 1, 1 + P.DOWNTURN["house_prices"])
    lgd_dt = E.lgd_profile(snap, st, ead[:, None], np.zeros((len(snap), 1)), z_dt, hpi_dt)[:, 0]
    coll = snap["collateral"].fillna(0).to_numpy(dtype=float)
    ctype = snap["collateral_type"].fillna("").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        sec = np.where(ead > 0, np.clip(coll * (1 - 0.4) / ead, 0, 1), 0)
    sec_floor = np.where(np.isin(ctype, ["CRE", "RRE"]), P.IRB_LGD_FLOOR["Corporate RRE/CRE"],
                         P.IRB_LGD_FLOOR["Corporate other physical"])
    corp_floor = np.where(ctype != "", sec * sec_floor + (1 - sec) * P.IRB_LGD_FLOOR["Corporate unsecured"],
                          P.IRB_LGD_FLOOR["Corporate unsecured"])
    lgd_floor = np.select([cls == "Corporate", cls == "Residential mortgage", cls == "QRRE"],
                          [corp_floor, P.IRB_LGD_FLOOR["Mortgage"], P.IRB_LGD_FLOOR["QRRE"]],
                          P.IRB_LGD_FLOOR["Other retail unsecured"])
    lgd_irb = np.maximum(lgd_dt, lgd_floor)
    firb_lgd = np.where(ctype != "", sec * np.where(np.isin(ctype, ["CRE", "RRE"]), 0.20, 0.25)
                        + (1 - sec) * P.FIRB_LGD_SENIOR, P.FIRB_LGD_SENIOR)
    lgd_irb = np.where(firb, firb_lgd, lgd_irb)
    m = np.where(firb, 2.5, np.clip(snap["remaining"].to_numpy(dtype=float), 1, 5))
    k, r = risk_weight_k(cls, np.where(dflt, 0.5, pd_irb), lgd_irb, m, snap["turnover"].to_numpy(dtype=float))
    elbe = res["lgd_default"].to_numpy()
    k = np.where(dflt, np.maximum(lgd_irb - elbe, 0), k)
    el = np.where(dflt, elbe * ead, pd_irb * lgd_irb * ead)
    out = pd.DataFrame({"id": snap["id"].to_numpy(), "segment": snap["segment"].to_numpy(), "irb_class": cls,
                        "defaulted": dflt, "stage": res["stage"].to_numpy(), "ead_irb": ead, "pd_irb": pd_irb,
                        "lgd_irb": lgd_irb, "maturity": m, "correlation": np.where(dflt, np.nan, r), "k": k,
                        "rwa": 12.5 * k * ead, "el": el, "provisions": res["ecl_total"].to_numpy(),
                        "pd12_ifrs9": res["pd12"].to_numpy(),
                        "lgd_ifrs9": np.where(dflt, elbe, np.nan)})
    return out


def shortfall(irb: pd.DataFrame, regime: str = "BCBS") -> dict:
    """Comparison of total EL with total eligible provisions (CRE35 / CRR Art. 159)."""
    nd, d = irb[~irb["defaulted"]], irb[irb["defaulted"]]
    el_nd, el_d = nd["el"].sum(), d["el"].sum()
    pr_nd, pr_d = nd["provisions"].sum(), d["provisions"].sum()
    rwa = irb["rwa"].sum()
    if regime.startswith("EU"):
        # specific provisions on defaulted exposures cannot cover EL on other exposures
        exc_nd, sh_nd = max(pr_nd - el_nd, 0), max(el_nd - pr_nd, 0)
        diff_d = pr_d - el_d
        if diff_d < 0:
            cover = min(exc_nd, -diff_d)
            sh = sh_nd + (-diff_d - cover)
            exc = exc_nd - cover
        else:
            sh, exc = sh_nd, exc_nd + diff_d
    else:
        diff = pr_nd + pr_d - el_nd - el_d
        sh, exc = max(-diff, 0), max(diff, 0)
    t2 = min(exc, P.T2_EXCESS_CAP * rwa)
    return {"EL non-defaulted": el_nd, "EL defaulted": el_d, "Provisions non-defaulted": pr_nd,
            "Provisions defaulted": pr_d, "Shortfall (deducted from CET1)": sh, "Excess": exc,
            "Excess eligible in Tier 2 (cap 0.6% credit RWA)": t2, "Credit RWA": rwa}
