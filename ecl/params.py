"""
Parameters of the IFRS 9 expected credit loss engine.

Sources
-------
* IFRS 9 Financial Instruments, section 5.5 and Appendix B (B5.5.1-B5.5.55):
  staging (5.5.3, 5.5.5), low credit risk (5.5.10, B5.5.22-24), 30 days past
  due rebuttable presumption (5.5.11, B5.5.19-20), 90 days default presumption
  (B5.5.37), probability-weighted forward-looking measurement (5.5.17-5.5.20).
* IFRS 7 Financial Instruments: Disclosures, paragraphs 35H-35I (reconciliation
  of loss allowance and gross carrying amount by stage).
* EBA/GL/2016/07 on the definition of default (probation periods) and
  EBA/GL/2017/06 on credit risk management practices and accounting for ECL.
* Basel Framework CRE31 (IRB risk weight functions) and CRE32 (risk components:
  PD floor 0.05% / 0.10% QRRE revolvers, F-IRB LGD 40% / 45% / 75%, A-IRB LGD
  floors), CRE35 (treatment of expected losses and provisions).

All numerical calibrations below are illustrative.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Rating scales (the last state is default, absorbing)
# --------------------------------------------------------------------------
SCALES = {
    "Corporate": ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"],
    "Retail": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "D"],
}
INVESTMENT_GRADE = {"AAA", "AA", "A", "BBB"}

# 1-year through-the-cycle default rates by grade (excluding D)
TTC_PD = {
    "Corporate": [0.0001, 0.0002, 0.0006, 0.0018, 0.0070, 0.0300, 0.2200],
    "Mortgage": [0.0010, 0.0025, 0.0050, 0.0100, 0.0250, 0.0800, 0.2500],
    "Consumer": [0.0040, 0.0080, 0.0150, 0.0300, 0.0600, 0.1400, 0.3500],
}
SEGMENT_SCALE = {"Corporate": "Corporate", "Mortgage": "Retail", "Consumer": "Retail"}

# Migration shape: probability of moving 1 and 2 notches up / down (before default)
MIGRATION_SHAPE = {
    "Corporate": dict(up1=0.050, up2=0.004, down1=0.060, down2=0.008),
    "Mortgage": dict(up1=0.080, up2=0.010, down1=0.060, down2=0.010),
    "Consumer": dict(up1=0.090, up2=0.015, down1=0.080, down2=0.020),
}

# Asset correlation used to condition transition matrices on the credit cycle Z
RHO = {"Corporate": 0.10, "Mortgage": 0.08, "Consumer": 0.05}


def build_ttc_matrix(segment: str) -> np.ndarray:
    """1-year TTC transition matrix with the target default column."""
    pd_ = TTC_PD[segment]
    sh = MIGRATION_SHAPE[segment]
    n = len(pd_)
    m = np.zeros((n + 1, n + 1))
    for i in range(n):
        for k, key in ((-1, "up1"), (-2, "up2"), (1, "down1"), (2, "down2")):
            j = i + k
            if 0 <= j < n:
                m[i, j] = sh[key] * (1 + 0.5 * i / n if k > 0 else 1.0)
        m[i, n] = pd_[i]
        m[i, i] = 1.0 - m[i].sum()
    m[n, n] = 1.0
    return m


def normalise_matrix(m: np.ndarray) -> np.ndarray:
    m = np.clip(np.asarray(m, dtype=float), 0.0, None)
    m = m / m.sum(axis=1, keepdims=True)
    m[-1] = 0.0
    m[-1, -1] = 1.0
    return m


# --------------------------------------------------------------------------
# Products
# --------------------------------------------------------------------------
PRODUCTS = {
    # segment, amortisation, behavioural CCF on undrawn (IFRS 9), annual CPR, behavioural life (revolving)
    "Corporate term loan": dict(segment="Corporate", amort="Linear", ccf=0.0, cpr=0.03, life=None),
    "Corporate RCF": dict(segment="Corporate", amort="Revolving", ccf=0.60, cpr=0.0, life=3),
    "Mortgage": dict(segment="Mortgage", amort="Annuity", ccf=0.0, cpr=0.06, life=None),
    "Consumer loan": dict(segment="Consumer", amort="Annuity", ccf=0.0, cpr=0.10, life=None),
    "Credit card": dict(segment="Consumer", amort="Revolving", ccf=0.50, cpr=0.0, life=3),
}

# LGD model parameters
LGD = {
    "Corporate": dict(unsecured=0.45, cure=0.20, haircut=0.40, sensitivity=0.04, cost=0.02),
    "Mortgage": dict(unsecured=0.60, cure=0.30, haircut=0.30, sensitivity=0.00, cost=0.10),
    "Consumer": dict(unsecured=0.70, cure=0.15, haircut=0.00, sensitivity=0.05, cost=0.03),
}
TIME_TO_SALE = 2.0                      # years from default to collateral sale (mortgages)

# --------------------------------------------------------------------------
# Macroeconomic scenarios (annual values for the next three years)
# --------------------------------------------------------------------------
MACRO_VARS = ["GDP growth (%)", "Unemployment (%)", "House prices (%)"]
LONG_RUN = {"GDP growth (%)": 1.3, "Unemployment (%)": 6.5, "House prices (%)": 2.0}
Z_LOADINGS = {"GDP growth (%)": (0.60, 1.5), "Unemployment (%)": (-0.40, 1.0)}   # (beta, std)
MEAN_REVERSION = 0.5                    # Z_t = Z_3 * phi^(t-3) beyond the 3-year horizon

SCENARIOS_CLOSE = {
    # name: (weight, {var: [y1, y2, y3]})
    "Upside": (0.15, {"GDP growth (%)": [2.0, 2.2, 1.9], "Unemployment (%)": [6.1, 5.9, 5.8],
                      "House prices (%)": [4.0, 4.0, 3.0]}),
    "Base": (0.50, {"GDP growth (%)": [1.1, 1.3, 1.4], "Unemployment (%)": [6.4, 6.3, 6.2],
                    "House prices (%)": [2.0, 2.5, 2.5]}),
    "Downside": (0.25, {"GDP growth (%)": [-0.8, 0.5, 1.2], "Unemployment (%)": [7.2, 7.6, 7.4],
                        "House prices (%)": [-4.0, -2.0, 1.0]}),
    "Severe": (0.10, {"GDP growth (%)": [-3.0, -0.5, 1.0], "Unemployment (%)": [8.5, 9.5, 9.0],
                      "House prices (%)": [-12.0, -6.0, 0.0]}),
}
# Scenarios used one year earlier (opening balance, for the IFRS 7 reconciliation)
SCENARIOS_OPEN = {
    "Upside": (0.20, {"GDP growth (%)": [2.2, 2.1, 1.8], "Unemployment (%)": [6.0, 5.8, 5.8],
                      "House prices (%)": [5.0, 4.0, 3.0]}),
    "Base": (0.55, {"GDP growth (%)": [1.5, 1.5, 1.4], "Unemployment (%)": [6.3, 6.2, 6.2],
                    "House prices (%)": [3.0, 3.0, 2.5]}),
    "Downside": (0.20, {"GDP growth (%)": [0.0, 0.8, 1.2], "Unemployment (%)": [7.0, 7.2, 7.0],
                        "House prices (%)": [-2.0, 0.0, 1.5]}),
    "Severe": (0.05, {"GDP growth (%)": [-2.5, 0.0, 1.0], "Unemployment (%)": [8.2, 9.0, 8.6],
                      "House prices (%)": [-10.0, -4.0, 0.0]}),
}
REALISED_Z = -0.30                      # credit cycle realised over the past year (demo generator)

# --------------------------------------------------------------------------
# Staging defaults
# --------------------------------------------------------------------------
SICR_DEFAULTS = dict(
    relative={"Corporate": 4.0, "Mortgage": 2.5, "Consumer": 2.5},  # lifetime PD now / at origination
    absolute_bp=50.0,                   # and an absolute increase of at least 50 bp
    backstop_30dpd=True,
    low_credit_risk=True,               # investment-grade corporate exposures stay in Stage 1
    watchlist=True,
    forborne=True,
    probation_s2_months=3,              # time in Stage 2 after the trigger disappears
    probation_s3_months=3,              # EBA/GL/2016/07 para. 71: at least 3 months
    probation_s3_restructured_months=12,  # para. 72: at least one year for distressed restructuring
)
MAX_HORIZON = 30                        # years

# --------------------------------------------------------------------------
# IRB (Basel Framework CRE31-CRE32)
# --------------------------------------------------------------------------
IRB_PD_FLOOR = {"Corporate": 0.0005, "Mortgage": 0.0005, "Consumer": 0.0005, "QRRE revolver": 0.0010}
IRB_LGD_FLOOR = {"Corporate unsecured": 0.25, "Corporate RRE/CRE": 0.10, "Corporate other physical": 0.15,
                 "Mortgage": 0.05, "QRRE": 0.50, "Other retail unsecured": 0.30}
FIRB_LGD_SENIOR = 0.40                  # non-financial corporates (CRE32.6)
FIRB_CCF_COMMITMENT = 0.40              # standardised CCF for commitments (CRE20)
DOWNTURN = dict(z=-2.0, house_prices=-0.25)   # downturn conditions for the downturn LGD
T2_EXCESS_CAP = 0.006                   # excess provisions eligible in Tier 2: 0.6% of credit RWA (CRE35)
