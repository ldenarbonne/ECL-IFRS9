"""
IFRS 9 page: staging, probability-weighted ECL, post-model adjustments,
sensitivities, IFRS 7 reconciliation, comparison with IRB expected loss and
PD model validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ecl import demo
from ecl import engine as E
from ecl import io as eio
from ecl import irb as IRB
from ecl import movements as MV
from ecl import params as P
from ecl import pd as PDM
from ecl import validation as VAL

C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#d6a12a", "#d03b3b"]
STAGE_COLOR = {1: "#1baf7a", 2: "#d6a12a", 3: "#d03b3b"}
SC_COLOR = {"Upside": C[2], "Base": C[0], "Downside": C[1], "Severe": C[5]}
FLAG_COLOR = {"Green": "rgba(12,163,12,0.25)", "Amber": "rgba(214,161,42,0.35)", "Red": "rgba(208,59,59,0.35)"}


def m(x: float) -> str:
    return f"{x:,.1f} m"


def h(df: pd.DataFrame) -> int:
    """Height showing every row of a small table."""
    return min(35 * (len(df) + 1) + 3, 900)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("📉 IFRS 9 ECL")
st.sidebar.caption("Staging · ECL · IFRS 7 · IRB EL · validation")
st.sidebar.subheader("Loan book")
src = st.sidebar.radio("Input", ["Demo bank (synthetic)", "Upload CSV"])
pf, two_dates = None, True
if src == "Upload CSV":
    up = st.sidebar.file_uploader("Portfolio (two-date format or closing snapshot)", type=["csv", "txt"])
    if up is not None:
        try:
            pf, two_dates = eio.to_portfolio(pd.read_csv(up, sep=None, engine="python"))
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Could not read the file: {exc}")
    else:
        st.sidebar.info("Download a template in the **Portfolio** tab, fill it and upload it.")
if pf is None:
    seed = st.sidebar.number_input("Random seed of the demo book", 1, 999, 1)
    pf = demo.generate(int(seed))
    two_dates = True

st.sidebar.subheader("Significant increase in credit risk")
rel = {seg: st.sidebar.number_input(f"Relative threshold – {seg} (× lifetime PD at origination)", 1.0, 20.0,
                                    float(P.SICR_DEFAULTS["relative"][seg]), 0.5) for seg in P.TTC_PD}
abs_bp = st.sidebar.number_input("Absolute threshold (bp of lifetime PD)", 0.0, 1000.0,
                                 P.SICR_DEFAULTS["absolute_bp"], 10.0)
c1, c2 = st.sidebar.columns(2)
backstop = c1.checkbox("30 dpd backstop", True)
lcr = c2.checkbox("Low credit risk", True, help="Investment-grade corporate exposures stay in Stage 1 (IFRS 9 5.5.10)")
watch = c1.checkbox("Watchlist → S2", True)
forb = c2.checkbox("Forborne → S2", True)
with st.sidebar.expander("Probation periods (months)"):
    pr2 = st.number_input("Stage 2 → Stage 1", 0, 24, P.SICR_DEFAULTS["probation_s2_months"])
    pr3 = st.number_input("Exit from default", 0, 24, P.SICR_DEFAULTS["probation_s3_months"])
    pr3r = st.number_input("Exit from default – distressed restructuring", 0, 36,
                           P.SICR_DEFAULTS["probation_s3_restructured_months"])
st.sidebar.subheader("IRB comparison")
corp_appr = st.sidebar.radio("Corporate approach", ["A-IRB", "F-IRB"], horizontal=True)
regime = st.sidebar.radio("Provision shortfall rule", ["BCBS (CRE35)", "EU (CRR Art. 159)"])

st.title("IFRS 9 Expected Credit Loss Engine")
st.caption("Amounts in EUR million · closing date = reporting date, opening date = one year earlier · "
           "synthetic data unless a file is uploaded")

tabs = st.tabs(["Overview", "Macro & PD", "Staging", "ECL detail", "Overlays & sensitivities", "IFRS 7",
                "IRB vs IFRS 9", "Validation", "Portfolio", "Methodology"])

# --------------------------------------------------------------------------
# Inputs held in other tabs (macro scenarios, matrices, overlays)
# --------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Forward-looking scenarios (reporting date)")
    w0, p0 = eio.scenarios_to_tables(P.SCENARIOS_CLOSE)
    a, b = st.columns([1, 3])
    w_tbl = a.data_editor(w0, hide_index=True, key="w_close", disabled=["scenario"])
    p_tbl = b.data_editor(p0, hide_index=True, key="p_close", disabled=["scenario", "variable"])
    scen_close = eio.tables_to_scenarios(w_tbl, p_tbl)
    tot_w = sum(v[0] for v in scen_close.values())
    if abs(tot_w - 1) > 1e-6:
        st.warning(f"Weights sum to {tot_w:.0%}; they are rescaled to 100%.")
    with st.expander("Scenarios used at the opening date (IFRS 7 reconciliation and PD backtest)"):
        wo0, po0 = eio.scenarios_to_tables(P.SCENARIOS_OPEN)
        x, y = st.columns([1, 3])
        wo_tbl = x.data_editor(wo0, hide_index=True, key="w_open", disabled=["scenario"])
        po_tbl = y.data_editor(po0, hide_index=True, key="p_open", disabled=["scenario", "variable"])
        scen_open = eio.tables_to_scenarios(wo_tbl, po_tbl)
    with st.expander("TTC transition matrices and asset correlations"):
        mats, rhos = {}, {}
        for seg in P.TTC_PD:
            scale = P.SCALES[P.SEGMENT_SCALE[seg]]
            st.markdown(f"**{seg}** (rows are normalised to 100%; the default row is absorbing)")
            mt = st.data_editor(pd.DataFrame(P.build_ttc_matrix(seg) * 100, index=scale, columns=scale).round(3),
                                key=f"mat_{seg}")
            mats[seg] = P.normalise_matrix(mt.to_numpy() / 100)
            rhos[seg] = st.number_input(f"ρ {seg}", 0.01, 0.5, P.RHO[seg], 0.01, key=f"rho_{seg}")

with tabs[4]:
    st.subheader("Post-model adjustments (overlays)")
    overlays = st.data_editor(E.default_overlays(), num_rows="dynamic", hide_index=True, key="overlays",
                              column_config={
                                  "segment": st.column_config.SelectboxColumn(options=["All"] + list(P.TTC_PD)),
                                  "stage": st.column_config.SelectboxColumn(options=["All", "Stage 1", "Stage 2", "Stage 3"]),
                                  "type": st.column_config.SelectboxColumn(options=["% of ECL", "Amount"])},
                              width="stretch")

settings = E.Settings(matrices=mats, rho=rhos)
settings.sicr.update(relative=rel, absolute_bp=abs_bp, backstop_30dpd=backstop, low_credit_risk=lcr,
                     watchlist=watch, forborne=forb, probation_s2_months=pr2, probation_s3_months=pr3,
                     probation_s3_restructured_months=pr3r)
S_JSON = eio.settings_to_json(settings)
SC_JSON, SO_JSON = eio.scenarios_to_json(scen_close), eio.scenarios_to_json(scen_open)

errors = eio.validate(pf)
if errors:
    for e in errors[:20]:
        st.error(e)
    st.stop()


@st.cache_data(show_spinner="Computing ECL…", max_entries=8)
def run_close(pf: pd.DataFrame, s_json: str, sc_json: str, overlays: pd.DataFrame):
    st_ = eio.settings_from_json(s_json)
    snap = demo.snapshot(pf, "close")
    res = E.apply_overlays(E.compute_ecl(snap, st_, eio.scenarios_from_json(sc_json)), overlays)
    return snap, res


@st.cache_data(show_spinner="Reconciling opening and closing allowances…", max_entries=4)
def run_movements(pf, s_json, so_json, sc_json, overlays):
    return MV.run(pf, eio.settings_from_json(s_json), eio.scenarios_from_json(so_json),
                  eio.scenarios_from_json(sc_json), overlays)


@st.cache_data(show_spinner="Running the SICR sensitivity…", max_entries=4)
def sicr_sweep(pf, s_json, sc_json):
    st_ = eio.settings_from_json(s_json)
    snap = demo.snapshot(pf, "close")
    sc = eio.scenarios_from_json(sc_json)
    base_rel = dict(st_.sicr["relative"])
    rows = []
    for f in [0.5, 1.0, 1.5, 2.0, 3.0]:
        st_.sicr["relative"] = {k: max(1.0, v * f) for k, v in base_rel.items()}
        r = E.compute_ecl(snap, st_, sc)
        rows.append({"threshold multiplier": f, "Stage 2 share": r.loc[r["stage"] == 2, "gross"].sum() / r["gross"].sum(),
                     "model ECL": r["ecl"].sum()})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Backtesting PDs…", max_entries=4)
def run_validation(pf, s_json, so_json, basis):
    s = VAL.observation_sample(pf, eio.settings_from_json(s_json), eio.scenarios_from_json(so_json), basis)
    return s, VAL.calibration(s), VAL.discrimination(s), VAL.psi(pf)


snap, res = run_close(pf, S_JSON, SC_JSON, overlays)
scen_names = list(scen_close)
tot_g, tot_e = res["gross"].sum(), res["ecl_total"].sum()

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tabs[0]:
    k = st.columns(5)
    k[0].metric("Gross carrying amount", m(tot_g))
    k[1].metric("Loss allowance", m(tot_e), f"of which overlays {m(res['pma'].sum())}", delta_color="off")
    k[2].metric("Coverage ratio", f"{tot_e / tot_g:.2%}")
    k[3].metric("Stage 2 share", f"{res.loc[res.stage == 2, 'gross'].sum() / tot_g:.1%}")
    k[4].metric("Stage 3 ratio (NPL)", f"{res.loc[res.stage == 3, 'gross'].sum() / tot_g:.1%}")
    summ = E.stage_summary(res)
    fmt = {c: ("{:.2%}" if "coverage" in c else "{:,.1f}") for c in summ.columns}
    st.dataframe(summ.style.format(fmt), width="stretch", height=h(summ))
    a, b = st.columns(2)
    fig = go.Figure()
    for s in (1, 2, 3):
        d = res[res.stage == s].groupby("segment")["ecl_total"].sum()
        fig.add_trace(go.Bar(x=d.index, y=d.values, name=f"Stage {s}", marker_color=STAGE_COLOR[s]))
    fig.update_layout(barmode="stack", title="Loss allowance by segment and stage", height=380, yaxis_title="EUR m",
                      margin=dict(t=50))
    a.plotly_chart(fig, width="stretch")
    sc_tot = [res[f"ECL | {s}"].sum() for s in scen_names]
    fig = go.Figure(go.Bar(x=scen_names, y=sc_tot, marker_color=[SC_COLOR.get(s, C[3]) for s in scen_names],
                           text=[f"{v:,.1f}" for v in sc_tot], textposition="outside"))
    fig.add_hline(y=res["ecl"].sum(), line_dash="dash", annotation_text="Probability-weighted (model)",
                  annotation_position="top left")
    fig.update_layout(title="Model ECL if 100% weighted to each scenario (staging held constant)", height=380,
                      yaxis_title="EUR m", margin=dict(t=50))
    b.plotly_chart(fig, width="stretch")
    st.caption("The probability-weighted ECL exceeds the base-scenario ECL because losses are convex in the "
               "macroeconomic factor (IFRS 9 B5.5.42): averaging scenarios captures the non-linearity.")

# --------------------------------------------------------------------------
# Macro & PD (outputs)
# --------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Credit cycle index Z and PD term structures")
    a, b = st.columns(2)
    fig = go.Figure()
    for s, (_, mac) in scen_close.items():
        z = PDM.z_forecast(mac)
        zz = list(z) + [z[-1] * P.MEAN_REVERSION ** i for i in range(1, 5)]
        fig.add_trace(go.Scatter(x=list(range(1, len(zz) + 1)), y=zz, name=s, line=dict(color=SC_COLOR.get(s))))
    fig.update_layout(title="Z by year (forecast 3 years, then reversion)", height=360, xaxis_title="Year",
                      yaxis_title="Z (σ units, > 0 = benign)", margin=dict(t=50))
    a.plotly_chart(fig, width="stretch")
    seg = b.selectbox("Segment", list(P.TTC_PD))
    scale = P.SCALES[P.SEGMENT_SCALE[seg]][:-1]
    grade = b.selectbox("Grade", scale, index=3)
    tables = PDM.pd_tables({seg: settings.matrices[seg]}, scen_close, {seg: settings.rho[seg]})
    gi = scale.index(grade)
    fig = go.Figure()
    for s in scen_names + ["TTC"]:
        fig.add_trace(go.Scatter(x=np.arange(1, P.MAX_HORIZON + 1), y=tables[seg][s][gi] * 100, name=s,
                                 line=dict(color=SC_COLOR.get(s, "black"), dash="dot" if s == "TTC" else None)))
    fig.update_layout(title=f"Cumulative PD – {seg} {grade}", height=360, xaxis_title="Year",
                      yaxis_title="%", margin=dict(t=50))
    b.plotly_chart(fig, width="stretch")
    pit = pd.DataFrame({s: tables[seg][s][:, 0] for s in scen_names + ["TTC"]}, index=scale)
    st.markdown("**1-year PD by grade and scenario**")
    st.dataframe(pit.style.format("{:.3%}"), width="stretch", height=h(pit))

# --------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Stage allocation and triggers")
    rs = res.groupby(["stage", "reason"]).agg(exposures=("id", "count"), gross=("gross", "sum"),
                                               ECL=("ecl_total", "sum")).reset_index()
    rs["coverage"] = rs["ECL"] / rs["gross"].where(rs["gross"] > 0)
    st.dataframe(rs.style.format({"gross": "{:,.1f}", "ECL": "{:,.2f}", "coverage": "{:.2%}"}), hide_index=True,
                 width="stretch", height=h(rs))
    st.caption("Triggers are tested in order (Stage 3, then Stage 2) and the first one met is reported. "
               "Stage 2 quantitative test: lifetime PD now ≥ threshold × lifetime PD expected at origination "
               "for the remaining term **and** absolute increase ≥ the bp threshold.")
    smp = res[res.stage < 3].sample(min(3000, (res.stage < 3).sum()), random_state=1)
    fig = go.Figure()
    for s in (1, 2):
        d = smp[smp.stage == s]
        fig.add_trace(go.Scattergl(x=d["pd_orig_life"], y=d["pd_life"], mode="markers", name=f"Stage {s}",
                                   marker=dict(color=STAGE_COLOR[s], size=5, opacity=0.6),
                                   text=d["id"] + " · " + d["reason"]))
    lim = [1e-4, 1]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="no change", line=dict(color="grey", dash="dot")))
    fig.update_layout(xaxis_type="log", yaxis_type="log", height=460, xaxis_title="Lifetime PD expected at origination",
                      yaxis_title="Lifetime PD at reporting date (probability-weighted)", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    if two_dates:
        mv = run_movements(pf, S_JSON, SO_JSON, SC_JSON, overlays)
        st.subheader("Stage transfers over the year (number of exposures)")
        st.dataframe(mv["transfers_count"], width="stretch")

# --------------------------------------------------------------------------
# ECL detail
# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("By product")
    g = res.groupby(["product", "stage"]).agg(n=("id", "count"), gross=("gross", "sum"), EAD=("ead", "sum"),
                                              PD12=("pd12", "mean"), ECL=("ecl_total", "sum")).reset_index()
    g["coverage"] = g["ECL"] / g["gross"].where(g["gross"] > 0)
    st.dataframe(g.style.format({"gross": "{:,.1f}", "EAD": "{:,.1f}", "PD12": "{:.2%}", "ECL": "{:,.2f}",
                                 "coverage": "{:.2%}"}), hide_index=True, width="stretch", height=h(g))
    st.subheader("Exposure explorer")
    a, b = st.columns([1, 2])
    choices = res.sort_values("ecl_total", ascending=False)["id"].head(500).tolist()
    lid = a.selectbox("Exposure (500 largest allowances)", choices)
    row = snap[snap["id"] == lid]
    r1 = res[res["id"] == lid].iloc[0]
    a.dataframe(pd.concat([row.T.iloc[:, 0].astype(str),
                           pd.Series({"stage": r1.stage, "trigger": r1.reason, "PD 12m": f"{r1.pd12:.2%}",
                                      "PD lifetime": f"{r1.pd_life:.2%}", "PD lifetime at orig.": f"{r1.pd_orig_life:.2%}",
                                      "ECL 12m": f"{r1.ecl12:,.4f}", "ECL lifetime": f"{r1.ecl_life:,.4f}",
                                      "ECL (stage)": f"{r1.ecl:,.4f}", "overlay": f"{r1.pma:,.4f}"})]).astype(str).rename("value"),
                width="stretch")
    if r1.stage == 3:
        b.info(f"Stage 3: ECL = EAD × LGD = {r1.ead:,.3f} × {r1.lgd_default:.1%} = {r1.ecl_default:,.3f} "
               "(PD = 100%, LGD probability-weighted across scenarios).")
    else:
        ex = E.explain(row.reset_index(drop=True), settings, scen_close)
        b.markdown("**Year-by-year build-up of lifetime ECL** (Stage 1 keeps year 1 only)")
        b.dataframe(ex.style.format({c: "{:.4%}" for c in ex.columns if c.startswith("mPD")} |
                                    {"LGD weighted": "{:.1%}", "DF": "{:.4f}", "EAD": "{:,.4f}",
                                     "ECL contribution": "{:,.5f}", "time of default (y)": "{:.2f}"}),
                    hide_index=True, width="stretch", height=h(ex))

# --------------------------------------------------------------------------
# Overlays & sensitivities (outputs)
# --------------------------------------------------------------------------
with tabs[4]:
    k = st.columns(3)
    k[0].metric("Modelled ECL", m(res["ecl"].sum()))
    k[1].metric("Post-model adjustments", m(res["pma"].sum()), f"{res['pma'].sum() / tot_e:.1%} of allowance",
                delta_color="off")
    k[2].metric("Total loss allowance", m(tot_e))
    st.caption("Overlays are allocated to exposures pro rata to modelled ECL within their scope. The ECB expects "
               "them to be documented, governed and eventually integrated into models.")
    st.subheader("Sensitivities")
    sens = pd.DataFrame({"Model ECL": [res[f"ECL | {s}"].sum() for s in scen_names]}, index=[f"100% {s}" for s in scen_names])
    sens.loc["Probability-weighted"] = res["ecl"].sum()
    sens.loc["All performing exposures in Stage 2 (lifetime ECL)"] = res["ecl_all_lifetime"].sum()
    sens["vs weighted"] = sens["Model ECL"] - res["ecl"].sum()
    st.dataframe(sens.style.format("{:,.1f}"), width="stretch", height=h(sens))
    sw = sicr_sweep(pf, S_JSON, SC_JSON)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sw["threshold multiplier"], y=sw["Stage 2 share"] * 100, name="Stage 2 share (%)",
                             line=dict(color=C[4])))
    fig.add_trace(go.Scatter(x=sw["threshold multiplier"], y=sw["model ECL"], name="Model ECL (EUR m)", yaxis="y2",
                             line=dict(color=C[0])))
    fig.update_layout(title="Sensitivity to the relative SICR threshold (× current thresholds)", height=380,
                      xaxis_title="Multiplier applied to relative thresholds", yaxis_title="Stage 2 share (%)",
                      yaxis2=dict(overlaying="y", side="right", title="EUR m"), margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------
# IFRS 7
# --------------------------------------------------------------------------
with tabs[5]:
    if not two_dates:
        st.info("The reconciliation needs the two-date format (opening and closing data).")
    else:
        mv = run_movements(pf, S_JSON, SO_JSON, SC_JSON, overlays)
        st.subheader("Loss allowance – reconciliation by stage (IFRS 7.35H)")
        st.dataframe(mv["allowance"].style.format("{:,.2f}"), width="stretch", height=h(mv["allowance"]))
        st.subheader("Gross carrying amount – reconciliation by stage (IFRS 7.35I)")
        st.dataframe(mv["gross"].style.format("{:,.1f}"), width="stretch", height=h(mv["gross"]))
        al = mv["allowance"]["Total"]
        fig = go.Figure(go.Waterfall(x=list(al.index), y=list(al.values),
                                     measure=["absolute"] + ["relative"] * (len(al) - 2) + ["total"],
                                     connector=dict(line=dict(color="grey"))))
        fig.update_layout(title="Loss allowance: opening to closing", height=420, margin=dict(t=50, b=120))
        st.plotly_chart(fig, width="stretch")
        st.markdown("**Stage transfers (gross carrying amount at opening, or at closing for new exposures)**")
        st.dataframe(mv["transfers_gross"].style.format("{:,.1f}"), width="stretch")
        st.caption(f"Control: opening + movements − closing = {MV.check(mv['allowance']):.2e} (allowance), "
                   f"{MV.check(mv['gross']):.2e} (gross). Transfers move the opening amounts between stages; "
                   "the effect of moving from 12-month to lifetime ECL sits in the remeasurement line.")
        st.download_button("Download the reconciliations (CSV)",
                           pd.concat({"allowance": mv["allowance"], "gross": mv["gross"]}).to_csv().encode(),
                           "ifrs7_reconciliation.csv", "text/csv")

# --------------------------------------------------------------------------
# IRB vs IFRS 9
# --------------------------------------------------------------------------
with tabs[6]:
    irb = IRB.compute(snap, res, settings, corp_appr)
    sf = IRB.shortfall(irb, regime)
    k = st.columns(4)
    k[0].metric("IRB expected loss", m(irb["el"].sum()))
    k[1].metric("IFRS 9 provisions", m(irb["provisions"].sum()))
    k[2].metric("Shortfall (CET1 deduction)", m(sf["Shortfall (deducted from CET1)"]))
    k[3].metric("Excess eligible in Tier 2", m(sf["Excess eligible in Tier 2 (cap 0.6% credit RWA)"]),
                f"total excess {m(sf['Excess'])}", delta_color="off")
    g = irb.groupby(["irb_class", "defaulted"]).agg(EAD=("ead_irb", "sum"), RWA=("rwa", "sum"), EL=("el", "sum"),
                                                     provisions=("provisions", "sum")).reset_index()
    g["RW density"] = g["RWA"] / g["EAD"]
    g["provisions − EL"] = g["provisions"] - g["EL"]
    st.dataframe(g.style.format({"EAD": "{:,.1f}", "RWA": "{:,.1f}", "EL": "{:,.2f}", "provisions": "{:,.2f}",
                                 "RW density": "{:.1%}", "provisions − EL": "{:,.2f}"}), hide_index=True, width="stretch",
                 height=h(g))
    st.dataframe(pd.Series(sf).rename("EUR m").to_frame().style.format("{:,.2f}"), width="stretch", height=h(pd.Series(sf)))
    by_stage = irb.groupby("stage")[["el", "provisions"]].sum()
    fig = go.Figure([go.Bar(x=[f"Stage {s}" for s in by_stage.index], y=by_stage["el"], name="IRB EL", marker_color=C[3]),
                     go.Bar(x=[f"Stage {s}" for s in by_stage.index], y=by_stage["provisions"], name="IFRS 9 allowance",
                            marker_color=C[0])])
    fig.update_layout(barmode="group", height=360, title="EL vs allowance by stage", yaxis_title="EUR m", margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")
    st.markdown("""
| | IFRS 9 ECL | IRB expected loss |
|---|---|---|
| PD | Point-in-time, forward-looking, scenario-weighted | Through-the-cycle 1-year, floored (0.05%, 0.10% QRRE revolvers) |
| Horizon | 12 months (Stage 1) or lifetime (Stages 2-3) | 1 year |
| LGD | Best estimate, scenario-dependent, discounted at the EIR | Downturn, floored (A-IRB) or supervisory (F-IRB) |
| Defaulted | Lifetime ECL with PD = 100% | ELBE; capital for unexpected loss in default = LGD − ELBE |
""")
    st.caption("Basel: total eligible provisions are compared with total EL. EU (CRR Art. 159): specific "
               "provisions on defaulted exposures cannot cover EL on other exposures. A shortfall is deducted "
               "from CET1; an excess is added to Tier 2 up to 0.6% of credit RWA.")

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
with tabs[7]:
    if not two_dates:
        st.info("Backtesting needs the two-date format (ratings at opening and default status at closing).")
    else:
        basis = st.radio("PD tested", ["PIT", "TTC"], horizontal=True,
                         help="PIT: 12-month probability-weighted PD at the opening date. TTC: long-run 1-year PD.")
        sample, cal, disc, (psi_d, psi_t) = run_validation(pf, S_JSON, SO_JSON, basis)
        st.subheader("Calibration by grade (one-year window)")
        st.dataframe(cal.style.format({"PD (avg)": "{:.3%}", "ODR": "{:.3%}", "binomial p": "{:.3f}",
                                       "Jeffreys p": "{:.3f}"})
                     .map(lambda v: f"background-color: {FLAG_COLOR.get(v, '')}" if v in FLAG_COLOR else "",
                          subset=["result"]), hide_index=True, width="stretch", height=h(cal))
        st.caption("H0: the PD is not underestimated. Binomial: P(X ≥ defaults | N, PD). Jeffreys: posterior "
                   "Beta(D + ½, N − D + ½) evaluated at the PD. Green ≥ 5%, amber 1-5%, red < 1%. Grades with fewer "
                   f"than {VAL.MIN_N} obligors are not tested.")
        seg = st.selectbox("Segment", sorted(cal["segment"].unique()), key="val_seg")
        d = cal[cal["segment"] == seg]
        fig = go.Figure([go.Bar(x=d["grade"], y=d["ODR"] * 100, name="Observed default rate", marker_color=C[1]),
                         go.Scatter(x=d["grade"], y=d["PD (avg)"] * 100, name="Predicted PD", mode="lines+markers",
                                    line=dict(color=C[0]))])
        fig.update_layout(height=360, yaxis_title="%", title=f"Predicted vs observed – {seg}", margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")
        a, b = st.columns(2)
        a.markdown("**Discriminatory power**")
        a.dataframe(disc.style.format({"AUC": "{:.3f}", "AUC 95% CI low": "{:.3f}", "AUC 95% CI high": "{:.3f}",
                                       "Gini": "{:.3f}"}), hide_index=True, width="stretch")
        b.markdown("**Population stability (grades at opening vs closing)**")
        b.dataframe(psi_t.style.format({"PSI": "{:.4f}"}), hide_index=True, width="stretch")
        st.caption("The demo PDs are well calibrated by construction (defaults are simulated with the same matrices); "
                   "upload your own data to test a real model. PSI < 0.10 stable, 0.10-0.25 moderate, > 0.25 significant.")

# --------------------------------------------------------------------------
# Portfolio
# --------------------------------------------------------------------------
with tabs[8]:
    st.subheader("Loan book")
    st.dataframe(pf.head(2000), width="stretch", height=380)
    st.caption(f"{len(pf):,} rows (first 2,000 shown). Status: Existing, New (originated in the year), "
               "Derecognised (repaid or matured), Written-off.")
    a, b = st.columns(2)
    a.download_button("Download the demo book – two-date format (CSV)", demo.generate(1).to_csv(index=False).encode(),
                      "ifrs9_portfolio_demo.csv", "text/csv")
    b.download_button("Download a closing snapshot template (CSV)",
                      demo.snapshot(demo.generate(1), "close").head(200).to_csv(index=False).encode(),
                      "ifrs9_snapshot_template.csv", "text/csv")
    st.markdown("""
| Field | Meaning |
|---|---|
| `segment`, `product` | Corporate (Corporate term loan, Corporate RCF), Mortgage, Consumer (Consumer loan, Credit card) |
| `amort` | Annuity, Linear, Bullet, Revolving |
| `orig_term`, `age`, `age_open` / `remaining` | Years; remaining term = orig_term − age |
| `limit`, `balance`, `balance_open` | Committed limit and drawn amount (EUR m) |
| `rate` | Effective interest rate (%), used for discounting |
| `rating_orig`, `rating_open`, `rating` | Grade at origination, opening and closing (AAA…CCC, D or R1…R7, D) |
| `dpd`, `utp`, `watchlist`, `forborne`, `restructured` | Days past due and qualitative flags |
| `cure_from`, `months_since_cure` | Last cure from S2 or S3 and months elapsed (probation) |
| `collateral`, `collateral_type` | Value (EUR m) and type: RRE, CRE, Other |
| `turnover` | Annual turnover (EUR m), for the IRB SME adjustment |
""")

# --------------------------------------------------------------------------
# Methodology
# --------------------------------------------------------------------------
with tabs[9]:
    st.markdown(r"""
### PD
* One 1-year TTC transition matrix per segment, conditioned on the credit cycle index $Z_t$ (Vasicek):
  $C_{ij}(Z)=\Phi\big((\Phi^{-1}(C_{ij})-\sqrt{\rho}Z)/\sqrt{1-\rho}\big)$ on cumulative row probabilities.
  Averaging over $Z\sim N(0,1)$ gives back the TTC matrix.
* $Z_t = 0.60\,(GDP_t-1.3)/1.5 - 0.40\,(U_t-6.5)/1.0$ over three years. Beyond that, the yearly matrix mixes
  the year-3 PIT matrix and the TTC matrix with weight $0.5^{t-3}$ (reversion to the long-run average).
* Lifetime PDs come from chaining yearly matrices; marginal PDs are interpolated log-linearly for partial years.

### EAD, LGD, discounting
* Monthly annuity or linear amortisation with a constant prepayment rate; bullet loans; revolving facilities
  with drawn + behavioural CCF × undrawn over a behavioural life (IFRS 9 5.5.20).
* LGD = (1 − cure rate) × (shortfall after collateral × unsecured loss rate + workout costs). Property values
  follow the house price path of each scenario up to the sale date; unsecured loss rates move with Z.
* Discounting at the effective interest rate to mid-year default times.

### Staging
* Stage 3: default grade, 90 dpd, unlikely to pay, or default probation (EBA/GL/2016/07: 3 months, 1 year after
  a distressed restructuring).
* Stage 2: relative and absolute lifetime PD increase vs origination, 30 dpd backstop (IFRS 9 5.5.11),
  watchlist, forbearance, Stage 2 probation; the low credit risk exemption (5.5.10) keeps investment-grade
  corporates in Stage 1.

### Limitations
* Synthetic data and illustrative calibration; the backtest is favourable by construction.
* One systematic factor per segment; no PD model by obligor characteristics.
* No purchased or originated credit-impaired (POCI) assets, no modification gains or losses, no interest
  revenue calculation on the net carrying amount for Stage 3.
* Staging held constant in single-scenario sensitivities.
* Educational tool: it does not replace an accounting or regulatory calculation.
""")

st.divider()
st.caption("Built by [Laurent Motais de Narbonne](https://www.linkedin.com/in/laurent-motais-de-narbonne/) · "
           "Sources: IFRS 9 §5.5 and App. B; IFRS 7 §35H-35I; EBA/GL/2016/07; EBA/GL/2017/06; Basel CRE31-32, CRE35.")
