"""IFRS 9 engine tests with hand-checked values."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta, binom, norm

from ecl import demo
from ecl import engine as E
from ecl import io as eio
from ecl import irb as IRB
from ecl import movements as MV
from ecl import params as P
from ecl import pd as PDM
from ecl import validation as VAL

NEUTRAL = {"Neutral": (1.0, {v: [P.LONG_RUN[v]] * 3 for v in P.MACRO_VARS})}   # Z = 0, house prices +2%/y


def loan(**kw):
    d = dict(id="X", segment="Corporate", product="Corporate term loan", amort="Bullet", balance=100.0, limit=100.0,
             rate=0.0, remaining=1.0, rating_orig="BBB", rating="BBB", dpd=0, utp=False, watchlist=False,
             forborne=False, restructured=False, cure_from="", months_since_cure=np.nan, collateral=np.nan,
             collateral_type="", turnover=np.nan)
    d.update(kw)
    return pd.DataFrame([d])[E.SNAP_COLUMNS]


# ---------------------------------------------------------------- PD
def test_ttc_matrices_are_stochastic_with_target_default_rates():
    for seg in P.TTC_PD:
        m = P.build_ttc_matrix(seg)
        assert np.allclose(m.sum(axis=1), 1)
        assert np.allclose(m[:-1, -1], P.TTC_PD[seg])
        assert m[-1, -1] == 1


def test_vasicek_default_column_and_cycle_average():
    m = P.build_ttc_matrix("Corporate")
    rho = 0.1
    c = PDM.conditional_matrix(m, -1.5, rho)
    expected = norm.cdf((norm.ppf(m[3, -1]) + np.sqrt(rho) * 1.5) / np.sqrt(1 - rho))
    assert c[3, -1] == pytest.approx(expected, rel=1e-10)
    assert np.allclose(c.sum(axis=1), 1)
    # averaging the PIT matrix over Z ~ N(0,1) gives back the TTC matrix
    z = np.linspace(-7, 7, 1401)
    w = norm.pdf(z) * (z[1] - z[0])
    avg = sum(wi * PDM.conditional_matrix(m, zi, rho) for wi, zi in zip(w, z))
    assert np.abs(avg - m).max() < 1e-6


def test_ttc_cumulative_pd_is_matrix_power():
    m = P.build_ttc_matrix("Mortgage")
    cum = PDM.cumulative_pd_ttc(m, 3)
    assert np.allclose(cum[:, 2], np.linalg.matrix_power(m, 3)[:-1, -1])


def test_reversion_to_ttc_beyond_forecast():
    m = P.build_ttc_matrix("Corporate")
    mats = PDM.yearly_matrices(m, np.array([-2.0, -2.0, -2.0]), 0.1, horizon=30)
    assert np.allclose(mats[-1], m, atol=1e-6)
    assert np.allclose(mats[3], 0.5 * mats[2] + 0.5 * m)


def test_z_from_macro():
    mac = {"GDP growth (%)": [-0.2, 1.3, 1.3], "Unemployment (%)": [7.5, 6.5, 6.5], "House prices (%)": [0, 0, 0]}
    z = PDM.z_forecast(mac)
    assert z[0] == pytest.approx(0.6 * (-1.5) / 1.5 - 0.4 * 1.0)
    assert z[1] == pytest.approx(0)


def test_log_linear_partial_year():
    cum = np.array([[0.19, 0.19 + 0.81 * 0.19]])           # constant 19% annual hazard
    assert E._cum_at(cum, np.array([[0.5]]))[0, 0] == pytest.approx(1 - np.sqrt(0.81))
    assert E._cum_at(cum, np.array([[1.5]]))[0, 0] == pytest.approx(1 - 0.81 * 0.9)


# ---------------------------------------------------------------- ECL
def test_one_year_bullet_stage1_hand_calculation():
    st = E.Settings()
    res = E.compute_ecl(loan(), st, NEUTRAL)
    pd1 = norm.cdf(norm.ppf(P.TTC_PD["Corporate"][3]) / np.sqrt(1 - P.RHO["Corporate"]))
    ead = 100 * (1 - 0.03) ** 0.5                              # prepayment survival to mid-year
    lgd = (1 - 0.20) * (1.0 * 0.45 + 0.02)
    assert res.loc[0, "stage"] == 1
    assert res.loc[0, "pd12"] == pytest.approx(pd1, rel=1e-9)
    assert res.loc[0, "ecl"] == pytest.approx(pd1 * ead * lgd, rel=1e-9)


def test_discounting_and_lifetime_two_years():
    st = E.Settings()
    st.products["Corporate term loan"]["cpr"] = 0.0
    r = E.compute_ecl(loan(remaining=2.0, rate=5.0), st, NEUTRAL)
    cum = PDM.cumulative_pd(st.matrices["Corporate"], np.zeros(3), P.RHO["Corporate"])[3]
    lgd = 0.8 * 0.47
    exp_life = 100 * lgd * (cum[0] / 1.05 ** 0.5 + (cum[1] - cum[0]) / 1.05 ** 1.5)
    assert r.loc[0, "ecl_life"] == pytest.approx(exp_life, rel=1e-9)
    assert r.loc[0, "ecl12"] == pytest.approx(100 * lgd * cum[0] / 1.05 ** 0.5, rel=1e-9)


def test_stage3_is_ead_times_lgd():
    st = E.Settings()
    r = E.compute_ecl(loan(rating="D", dpd=120), st, NEUTRAL)
    assert r.loc[0, "stage"] == 3
    assert r.loc[0, "ecl"] == pytest.approx(100 * 0.8 * 0.47)


def test_mortgage_lgd_uses_collateral_and_house_prices():
    st = E.Settings()
    d = loan(segment="Mortgage", product="Mortgage", amort="Annuity", balance=100, limit=100, rating="D",
             rating_orig="R3", remaining=20, rate=0.0, collateral=100.0, collateral_type="RRE")
    r = E.compute_ecl(d, st, NEUTRAL)
    rec = 100 * 1.02 ** 2 * (1 - 0.30)                        # house prices +2%/y over the 2-year sale delay
    exp = (1 - 0.30) * ((100 - rec) / 100 * 0.60 + 0.10)
    assert r.loc[0, "lgd_default"] == pytest.approx(exp)


def test_probability_weighting():
    st = E.Settings()
    d = demo.snapshot(demo.generate(3), "close").head(300)
    r = E.compute_ecl(d, st, P.SCENARIOS_CLOSE)
    w = {k: v[0] for k, v in P.SCENARIOS_CLOSE.items()}
    assert r["ecl"].sum() == pytest.approx(sum(w[s] * r[f"ECL | {s}"].sum() for s in w))
    assert r["ECL | Severe"].sum() > r["ECL | Base"].sum() > r["ECL | Upside"].sum()


def test_revolving_ead_uses_ccf():
    st = E.Settings()
    d = loan(product="Corporate RCF", amort="Revolving", balance=40, limit=100, remaining=0.5)
    ead = E.ead_profile(d, st, np.array([[0.0, 2.0]]))
    assert np.allclose(ead, 40 + 0.6 * 60)


# ---------------------------------------------------------------- staging
def stage_of(**kw):
    r = E.compute_ecl(loan(**kw), E.Settings(), NEUTRAL)
    return int(r.loc[0, "stage"]), r.loc[0, "reason"]


def test_staging_triggers():
    assert stage_of() == (1, "Performing")
    assert stage_of(dpd=35) == (2, "30 days past due")
    assert stage_of(dpd=95) == (3, "90 days past due")
    assert stage_of(utp=True) == (3, "Unlikely to pay")
    assert stage_of(watchlist=True) == (2, "Watchlist")
    assert stage_of(rating_orig="BBB", rating="B", remaining=5) == (2, "PD deterioration")
    assert stage_of(cure_from="S3", months_since_cure=2) == (3, "Default probation")
    assert stage_of(cure_from="S3", months_since_cure=2, restructured=True)[0] == 3
    assert stage_of(cure_from="S3", months_since_cure=10, restructured=True) == (3, "Default probation")
    assert stage_of(cure_from="S3", months_since_cure=4) == (2, "Post-default probation")
    assert stage_of(cure_from="S2", months_since_cure=1) == (2, "Stage 2 probation")


def test_low_credit_risk_exemption():
    # AA -> BBB is a large relative increase but BBB is investment grade
    assert stage_of(rating_orig="AA", rating="BBB", remaining=5) == (1, "Low credit risk exemption")
    st = E.Settings()
    st.sicr["low_credit_risk"] = False
    r = E.compute_ecl(loan(rating_orig="AA", rating="BBB", remaining=5), st, NEUTRAL)
    assert r.loc[0, "stage"] == 2


def test_absolute_threshold_blocks_tiny_increases():
    st = E.Settings()
    st.sicr["low_credit_risk"] = False
    st.sicr["absolute_bp"] = 1000
    r = E.compute_ecl(loan(rating_orig="AA", rating="BBB", remaining=5), st, NEUTRAL)
    assert r.loc[0, "stage"] == 1


# ---------------------------------------------------------------- overlays, movements
def test_overlays_allocation():
    res = pd.DataFrame({"id": ["a", "b", "c"], "segment": ["Corporate", "Corporate", "Mortgage"],
                        "stage": [1, 2, 2], "ecl": [1.0, 3.0, 2.0]})
    ov = pd.DataFrame([("Corporate", "All", "% of ECL", 10.0, ""), ("All", "Stage 2", "Amount", 5.0, "")],
                      columns=E.OVERLAY_COLUMNS)
    out = E.apply_overlays(res, ov)
    assert out["pma"].tolist() == pytest.approx([0.1, 0.3 + 3.0, 2.0])
    assert out["ecl_total"].sum() == pytest.approx(6.0 + 0.4 + 5.0)


def test_ifrs7_reconciliation_closes():
    pf = demo.generate(2)
    pf = pf.groupby("product").head(150).reset_index(drop=True)
    mv = MV.run(pf, E.Settings(), P.SCENARIOS_OPEN, P.SCENARIOS_CLOSE, E.default_overlays())
    assert MV.check(mv["allowance"]) < 1e-9 and MV.check(mv["gross"]) < 1e-9
    for t in (1, 2, 3):
        assert mv["allowance"].loc[f"Transfer to Stage {t}", "Total"] == pytest.approx(0, abs=1e-9)
    assert mv["allowance"].loc["Write-offs", ["Stage 1", "Stage 2"]].abs().sum() == 0
    closing = E.apply_overlays(E.compute_ecl(demo.snapshot(pf, "close"), E.Settings(), P.SCENARIOS_CLOSE),
                               E.default_overlays())
    assert mv["allowance"].loc["Closing balance", "Total"] == pytest.approx(closing["ecl_total"].sum())


# ---------------------------------------------------------------- IRB
def test_irb_risk_weights_match_basel_reference():
    # Basel II reference values (LGD 45%, M = 2.5, before the 1.06 scaling): corporate PD 1% -> 92.32%,
    # residential mortgage PD 1% -> 56.40%
    k, _ = IRB.risk_weight_k(np.array(["Corporate", "Residential mortgage"]), np.array([0.01, 0.01]),
                             np.array([0.45, 0.45]), np.array([2.5, 2.5]), np.array([np.nan, np.nan]))
    assert 12.5 * k[0] == pytest.approx(0.9232, abs=5e-4)
    assert 12.5 * k[1] == pytest.approx(0.5640, abs=5e-4)
    # SME adjustment lowers the correlation by up to 0.04
    _, r_sme = IRB.risk_weight_k(np.array(["Corporate"]), np.array([0.01]), np.array([0.45]), np.array([2.5]),
                                 np.array([5.0]))
    _, r = IRB.risk_weight_k(np.array(["Corporate"]), np.array([0.01]), np.array([0.45]), np.array([2.5]),
                             np.array([np.nan]))
    assert r - r_sme == pytest.approx(0.04)


def test_irb_floors_and_shortfall_rules():
    pf = demo.generate(1)
    snap = demo.snapshot(pf, "close")
    res = E.apply_overlays(E.compute_ecl(snap, E.Settings(), P.SCENARIOS_CLOSE), E.default_overlays())
    irb = IRB.compute(snap, res, E.Settings())
    perf = ~irb["defaulted"]
    assert irb.loc[perf & (irb["irb_class"] == "QRRE"), "pd_irb"].min() >= 0.001
    assert irb.loc[perf, "pd_irb"].min() >= 0.0005
    assert irb.loc[irb["irb_class"] == "Residential mortgage", "lgd_irb"].min() >= 0.05
    assert irb.loc[irb["irb_class"] == "QRRE", "lgd_irb"].min() >= 0.50
    toy = pd.DataFrame({"defaulted": [False, True], "el": [10.0, 5.0], "provisions": [6.0, 12.0], "rwa": [1000.0, 0.0]})
    bc, eu = IRB.shortfall(toy, "BCBS"), IRB.shortfall(toy, "EU")
    assert bc["Shortfall (deducted from CET1)"] == 0 and bc["Excess"] == pytest.approx(3)
    assert eu["Shortfall (deducted from CET1)"] == pytest.approx(4) and eu["Excess"] == pytest.approx(7)
    assert eu["Excess eligible in Tier 2 (cap 0.6% credit RWA)"] == pytest.approx(6)


# ---------------------------------------------------------------- validation
def test_validation_statistics():
    assert VAL.auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert VAL.auc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
    s = pd.DataFrame({"segment": "X", "grade": "G", "pd": 0.02, "default": [True] * 5 + [False] * 95})
    row = VAL.calibration(s).iloc[0]
    assert row["binomial p"] == pytest.approx(binom.sf(4, 100, 0.02))
    assert row["Jeffreys p"] == pytest.approx(beta.cdf(0.02, 5.5, 95.5))
    assert row["result"] == "Amber"                           # Jeffreys p = 2.9%
    s.loc[5, "default"] = True
    assert VAL.calibration(s).iloc[0]["result"] == "Red"     # 6 defaults: Jeffreys p = 0.8%


def test_demo_backtest_and_psi():
    pf = demo.generate(1)
    s = VAL.observation_sample(pf, E.Settings(), P.SCENARIOS_OPEN)
    disc = VAL.discrimination(s)
    assert (disc["AUC"] > 0.75).all()
    _, tot = VAL.psi(pf)
    assert (tot["PSI"] < 0.10).all()


# ---------------------------------------------------------------- io and app
def test_io_roundtrips():
    pf = demo.generate(1)
    p2, two = eio.to_portfolio(pf.astype({"utp": str}))
    assert two and eio.validate(p2) == []
    snap = demo.snapshot(pf, "close")
    p3, two = eio.to_portfolio(snap)
    assert not two
    pd.testing.assert_frame_equal(demo.snapshot(p3, "close").drop(columns="rating_orig"),
                                  snap.drop(columns="rating_orig"), check_dtype=False)
    st = eio.settings_from_json(eio.settings_to_json(E.Settings()))
    assert np.allclose(st.matrices["Consumer"], P.build_ttc_matrix("Consumer"))
    w, paths = eio.scenarios_to_tables(P.SCENARIOS_CLOSE)
    assert eio.tables_to_scenarios(w, paths) == P.SCENARIOS_CLOSE
    bad = pf.head(3).copy()
    bad.loc[bad.index[0], "product"] = "Leasing"
    assert eio.validate(bad)


def test_app_runs():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=900).run()
    assert not at.exception and not at.error
    assert at.metric[0].label == "Gross carrying amount"
    next(r for r in at.radio if r.label == "Corporate approach").set_value("F-IRB").run()
    next(r for r in at.radio if r.label == "Provision shortfall rule").set_value("EU (CRR Art. 159)").run()
    assert not at.exception and not at.error
