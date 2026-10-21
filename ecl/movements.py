"""
IFRS 7 paragraph 35H-35I reconciliations of the loss allowance and of the
gross carrying amount by stage between the opening and the closing date.

Conventions (one common presentation among several used in practice):
* transfers move the opening allowance / gross amount of an exposure from its
  opening stage to its closing stage (zero in total);
* the remeasurement of existing exposures is booked in the closing stage and
  split between the change in forward-looking information (closing ECL under
  closing vs opening scenarios, staging held constant) and all other effects
  (credit quality, 12-month vs lifetime switch, amortisation, unwinding);
* new exposures enter at their closing ECL; derecognised and written-off
  exposures leave at their opening ECL;
* post-model adjustments are shown on a separate line (nil at opening).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import demo
from . import engine as E

STAGES = [1, 2, 3]


def _row(label: str, values_by_stage: dict) -> dict:
    d = {"Movement": label}
    for s in STAGES:
        d[f"Stage {s}"] = float(values_by_stage.get(s, 0.0))
    d["Total"] = sum(d[f"Stage {s}"] for s in STAGES)
    return d


def run(portfolio: pd.DataFrame, st: E.Settings, scen_open: dict, scen_close: dict,
        overlays: pd.DataFrame) -> dict:
    o = demo.snapshot(portfolio, "open")
    c = demo.snapshot(portfolio, "close")
    r_o = E.compute_ecl(o, st, scen_open)
    r_c = E.apply_overlays(E.compute_ecl(c, st, scen_close), overlays)
    r_cm = E.compute_ecl(c, st, scen_open, force_stage=r_c["stage"].to_numpy())
    status = portfolio.set_index("id")["status"]
    a = r_o[["id", "stage", "ecl", "gross"]].rename(columns={"stage": "s_o", "ecl": "e_o", "gross": "g_o"})
    b = r_c[["id", "stage", "ecl", "pma", "ecl_total", "gross"]].rename(
        columns={"stage": "s_c", "ecl": "e_c", "gross": "g_c"})
    b["e_cm"] = r_cm["ecl"].to_numpy()
    m = a.merge(b, on="id", how="outer")
    m["status"] = m["id"].map(status)
    ex = m[m["status"] == "Existing"]

    def by(df, stage_col, val):
        return df.groupby(stage_col)[val].sum().to_dict() if len(df) else {}

    allow, gross = [], []
    allow.append(_row("Opening balance", by(m.dropna(subset=["s_o"]), "s_o", "e_o")))
    gross.append(_row("Opening balance", by(m.dropna(subset=["s_o"]), "s_o", "g_o")))
    for tgt in STAGES:
        mv = ex[(ex["s_c"] == tgt) & (ex["s_o"] != tgt)]
        for rows, val in ((allow, "e_o"), (gross, "g_o")):
            v = {s: -mv.loc[mv["s_o"] == s, val].sum() for s in STAGES}
            v[tgt] = v.get(tgt, 0) + mv[val].sum()
            rows.append(_row(f"Transfer to Stage {tgt}", v))
    allow.append(_row("Remeasurement (credit quality, stage switch)",
                      {s: (ex.loc[ex["s_c"] == s, "e_cm"] - ex.loc[ex["s_c"] == s, "e_o"]).sum() for s in STAGES}))
    allow.append(_row("Changes in forward-looking information",
                      {s: (ex.loc[ex["s_c"] == s, "e_c"] - ex.loc[ex["s_c"] == s, "e_cm"]).sum() for s in STAGES}))
    gross.append(_row("Repayments, drawdowns and other changes",
                      {s: (ex.loc[ex["s_c"] == s, "g_c"] - ex.loc[ex["s_c"] == s, "g_o"]).sum() for s in STAGES}))
    new = m[m["status"] == "New"]
    allow.append(_row("New financial assets originated", by(new, "s_c", "e_c")))
    gross.append(_row("New financial assets originated", by(new, "s_c", "g_c")))
    der = m[m["status"] == "Derecognised"]
    allow.append(_row("Financial assets derecognised (repaid, matured)", {k: -v for k, v in by(der, "s_o", "e_o").items()}))
    gross.append(_row("Financial assets derecognised (repaid, matured)", {k: -v for k, v in by(der, "s_o", "g_o").items()}))
    wo = m[m["status"] == "Written-off"]
    allow.append(_row("Write-offs", {k: -v for k, v in by(wo, "s_o", "e_o").items()}))
    gross.append(_row("Write-offs", {k: -v for k, v in by(wo, "s_o", "g_o").items()}))
    cl = m.dropna(subset=["s_c"])
    allow.append(_row("Post-model adjustments", by(cl, "s_c", "pma")))
    allow.append(_row("Closing balance", by(cl, "s_c", "ecl_total")))
    gross.append(_row("Closing balance", by(cl, "s_c", "g_c")))

    def lab(s):
        return np.where(m[s].isna(), None, "Stage " + m[s].fillna(0).astype(int).astype(str))
    m["from"] = np.where(m["status"] == "New", "New", lab("s_o"))
    m["to"] = np.where(m["status"] == "Derecognised", "Derecognised",
                       np.where(m["status"] == "Written-off", "Written-off", lab("s_c")))
    transfers = pd.crosstab(m["from"], m["to"], values=m["g_o"].fillna(m["g_c"]), aggfunc="sum").fillna(0)
    counts = pd.crosstab(m["from"], m["to"])
    return {"allowance": pd.DataFrame(allow).set_index("Movement"),
            "gross": pd.DataFrame(gross).set_index("Movement"),
            "transfers_gross": transfers, "transfers_count": counts, "detail": m,
            "open": r_o, "close": r_c}


def check(table: pd.DataFrame) -> float:
    """Difference between closing balance and opening + movements (should be ~0)."""
    body = table.drop(index=["Opening balance", "Closing balance"]).sum()
    return float((table.loc["Opening balance"] + body - table.loc["Closing balance"]).abs().max())
