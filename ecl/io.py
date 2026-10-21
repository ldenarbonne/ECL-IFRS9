"""CSV import, validation and serialisation of settings for caching."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import demo
from . import engine as E
from . import params as P

BOOL_COLS = ["utp_open", "utp", "watchlist_open", "watchlist", "forborne_open", "forborne", "restructured"]
NUM_COLS = ["orig_term", "age_open", "age", "limit", "balance_open", "balance", "rate", "dpd_open", "dpd",
            "months_since_cure", "collateral_open", "collateral", "turnover"]


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def to_portfolio(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Accept either the two-date portfolio format or a closing snapshot (engine.SNAP_COLUMNS).
    Returns (portfolio, has_opening_data)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    two_dates = {"status", "rating_open", "balance_open"}.issubset(df.columns)
    if not two_dates:
        if "remaining" not in df.columns:
            raise ValueError("Missing column 'remaining' (closing snapshot) or 'status' (two-date format)")
        df["orig_term"] = pd.to_numeric(df["remaining"], errors="coerce")
        df["age"] = 0.0
        df["status"] = "Existing"
    for c in demo.PORTFOLIO_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in BOOL_COLS:
        df[c] = _bool(df[c])
    for c in ["id", "segment", "product", "amort", "status", "rating_orig", "rating_open", "rating", "cure_from",
              "collateral_type"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["amort"] = np.where(df["amort"] == "", df["product"].map(lambda p: P.PRODUCTS.get(p, {}).get("amort", "")),
                           df["amort"])
    df["limit"] = df["limit"].fillna(df["balance"])
    for c in ["dpd", "dpd_open"]:
        df[c] = df[c].fillna(0)
    df["rating_orig"] = np.where(df["rating_orig"] == "", df["rating"], df["rating_orig"])
    return df[demo.PORTFOLIO_COLUMNS], two_dates


def validate(pf: pd.DataFrame) -> list[str]:
    errs = []
    for i, r in pf.iterrows():
        tag = f"Row {i + 1} ({r['id']})"
        if r["product"] not in P.PRODUCTS:
            errs.append(f"{tag}: unknown product '{r['product']}' (expected {list(P.PRODUCTS)})")
            continue
        if r["segment"] != P.PRODUCTS[r["product"]]["segment"]:
            errs.append(f"{tag}: product '{r['product']}' belongs to segment {P.PRODUCTS[r['product']]['segment']}")
        scale = P.SCALES[P.SEGMENT_SCALE[P.PRODUCTS[r["product"]]["segment"]]]
        if r["status"] in ("Existing", "New") and r["rating"] not in scale:
            errs.append(f"{tag}: rating '{r['rating']}' not in scale {scale}")
        if r["rating_orig"] not in scale:
            errs.append(f"{tag}: origination rating '{r['rating_orig']}' not in scale")
        if r["status"] not in ("Existing", "New", "Derecognised", "Written-off"):
            errs.append(f"{tag}: unknown status '{r['status']}'")
        if r["status"] in ("Existing", "New") and not (r["balance"] >= 0):
            errs.append(f"{tag}: balance must be a non-negative number")
        if len(errs) > 50:
            errs.append("…")
            break
    return errs


def settings_to_json(st: E.Settings) -> str:
    return json.dumps({"matrices": {k: np.asarray(v).tolist() for k, v in st.matrices.items()},
                       "rho": st.rho, "sicr": st.sicr, "products": st.products, "lgd": st.lgd}, sort_keys=True)


def settings_from_json(s: str) -> E.Settings:
    d = json.loads(s)
    return E.Settings(matrices={k: np.asarray(v) for k, v in d["matrices"].items()}, rho=d["rho"], sicr=d["sicr"],
                      products=d["products"], lgd=d["lgd"])


def scenarios_to_tables(sc: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = pd.DataFrame({"scenario": list(sc), "weight (%)": [v[0] * 100 for v in sc.values()]})
    rows = []
    for name, (_, mac) in sc.items():
        for var, vals in mac.items():
            rows.append({"scenario": name, "variable": var, **{f"Y{i + 1}": v for i, v in enumerate(vals)}})
    return w, pd.DataFrame(rows)


def tables_to_scenarios(w: pd.DataFrame, paths: pd.DataFrame) -> dict:
    out = {}
    for _, r in w.dropna(subset=["scenario"]).iterrows():
        sub = paths[paths["scenario"] == r["scenario"]]
        mac = {v: [float(x) for x in sub.loc[sub["variable"] == v, ["Y1", "Y2", "Y3"]].iloc[0]]
               for v in P.MACRO_VARS if (sub["variable"] == v).any()}
        for v in P.MACRO_VARS:
            mac.setdefault(v, [P.LONG_RUN[v]] * 3)
        out[str(r["scenario"])] = (float(r["weight (%)"]) / 100, mac)
    return out


def scenarios_to_json(sc: dict) -> str:
    return json.dumps(sc, sort_keys=True)


def scenarios_from_json(s: str) -> dict:
    return {k: (v[0], v[1]) for k, v in json.loads(s).items()}
