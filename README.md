# Credit Risk Suite – IFRS 9 Expected Credit Loss Engine

An interactive Streamlit application that computes **IFRS 9 expected credit losses** on a retail and corporate loan book. It covers:

* **SICR staging**: quantitative, qualitative, backstops and probation periods.
* **Probability-weighted ECL**: forward-looking point-in-time PDs, scenario LGDs, EAD profiles and discounting at the effective interest rate.
* **Post-model adjustments and sensitivities.**
* **IFRS 7 reconciliations**: loss allowance and gross carrying amount by stage.
* **Comparison with IRB expected loss**: shortfall or excess.
* **PD model validation**: calibration, discriminatory power and population stability.

The repository is organised as a multipage suite (`app.py` router, `views/`), so that further credit modules (IRB capital, credit stress testing) can be added as new pages.

**Live app:** `https://<your-app-name>.streamlit.app` *(to be completed after deployment)*

![tests](https://github.com/ldenarbonne/credit-risk-suite/actions/workflows/tests.yml/badge.svg)

---

## 1. Features

| Tab | Content |
|---|---|
| **Overview** | Gross carrying amount, loss allowance, coverage, Stage 2 share, NPL ratio. Stage table by segment. ECL if 100% weighted to each scenario |
| **Macro & PD** | Editable scenarios (weights, GDP, unemployment, house prices, 3 years) for the closing and opening dates. Credit cycle index Z. Editable TTC transition matrices and correlations. PD term structures by grade and scenario |
| **Staging** | Allocation by trigger, lifetime PD now vs at origination (scatter), stage transfers over the year |
| **ECL detail** | Results by product and stage. Exposure explorer with the year-by-year build-up of lifetime ECL (marginal PD per scenario, EAD, LGD, discount factor) |
| **Overlays & sensitivities** | Editable post-model adjustments (by segment and stage, % or amount). ECL under each scenario. All performing exposures in Stage 2. Sensitivity to the SICR threshold |
| **IFRS 7** | Reconciliation of the loss allowance (35H) and of the gross carrying amount (35I) by stage, waterfall, transfer matrix |
| **IRB vs IFRS 9** | IRB EAD, PD, LGD, RWA and EL (A-IRB or F-IRB for corporates). Comparison with provisions under the BCBS (CRE35) or EU (CRR Art. 159) rule. Shortfall deducted from CET1, excess added to Tier 2 up to 0.6% of credit RWA |
| **Validation** | Binomial and Jeffreys tests by grade, predicted vs observed default rates, AUC / Gini with confidence interval, PSI |
| **Portfolio** | Loan book, CSV templates, field reference |
| **Methodology** | Formulas, assumptions, limitations |

Inputs come from three sources:
* a synthetic bank of about 11,600 exposures observed at two dates (seed configurable);
* an uploaded CSV in the same two-date format;
* an uploaded closing snapshot only. In that case the IFRS 7 and backtesting tabs are disabled.

## 2. Method

### PD: transition matrices conditioned with Vasicek
* **Matrices**: one 1-year TTC matrix per segment (corporate AAA–CCC, retail R1–R7).
* **Conditioning on the cycle**: each row is conditioned on a credit cycle index Z with $C_{ij}(Z)=\Phi\big((\Phi^{-1}(C_{ij})-\sqrt{\rho}Z)/\sqrt{1-\rho}\big)$, applied to cumulative probabilities from default upwards. Averaging over Z ~ N(0,1) gives back the TTC matrix (this property is tested).
* **Z from the macro scenarios**: $Z_t = 0.60\,(GDP_t-1.3)/1.5 - 0.40\,(U_t-6.5)/1.0$ over three years.
* **Reversion to the long run**: the yearly matrix mixes the year-3 PIT matrix with the TTC matrix (weight $0.5^{t-3}$). PDs therefore revert to the long-run average, not to the median scenario.
* **Lifetime PDs**: yearly matrices are chained, with log-linear interpolation for partial years.

### EAD, LGD and ECL
* **EAD**:
  * monthly annuity, linear or bullet amortisation with prepayments;
  * revolving facilities: drawn amount + behavioural CCF × undrawn amount, over a behavioural life (IFRS 9 5.5.20).
* **LGD** = (1 − cure rate) × (shortfall after collateral × unsecured loss rate + workout costs):
  * property values follow the house price path of each scenario until the sale date;
  * unsecured loss rates move with Z.
* **ECL**: $\text{ECL}_s=\sum_t mPD_{s,t}\,EAD_t\,LGD_{s,t}\,DF_t$, discounted at the EIR to the middle of each year.
  * Stage 1 keeps year 1 only, Stage 2 the whole remaining life.
  * Stage 3: ECL = EAD × LGD (PD = 100%).
  * The reported figure is the probability-weighted average across scenarios.

### Staging
| Stage | Triggers (tested in order) |
|---|---|
| 3 | Default grade, 90 days past due, unlikely to pay, default probation (3 months, 12 months after a distressed restructuring – EBA/GL/2016/07) |
| 2 | Lifetime PD now ≥ k × lifetime PD expected at origination for the remaining term **and** increase ≥ 50 bp (k = 4 corporate, 2.5 retail); 30 days past due backstop; watchlist; forbearance; Stage 2 probation (3 months); post-default probation |
| 1 | Otherwise. Low credit risk exemption for investment-grade corporates (IFRS 9 5.5.10) |

## 3. Illustrative results (demo bank, seed 1, EUR m)

| | Stage 1 | Stage 2 | Stage 3 | Total |
|---|---|---|---|---|
| Gross carrying amount | 2,787 | 343 | 78 | 3,209 |
| Loss allowance (incl. overlays) | 8.0 | 21.5 | 17.1 | 46.6 |
| Coverage | 0.29% | 6.3% | 21.8% | 1.45% |

| Sensitivity | Model ECL |
|---|---|
| 100% Upside / Base / Downside / Severe | 28.7 / 33.7 / 48.9 / 84.2 |
| Probability-weighted (15/50/25/10) | 41.8 |
| All performing exposures in Stage 2 | 56.8 |

* **Probability weighting**: the weighted ECL exceeds the base case by 24%. This is the non-linearity that IFRS 9 B5.5.42 requires to be captured.
* **Allowance over the year**: it rises from 23.5 to 46.6. The increase splits into:
  * credit quality and stage changes: +12.0;
  * forward-looking information: +5.4;
  * new loans: +5.1;
  * overlays: +4.8;
  * write-offs: −3.4.
* **IRB**: credit RWA are 1,373 (A-IRB) and EL is 32.1, against 46.6 of provisions. The resulting excess of 14.5 is eligible in Tier 2 up to 8.2 (0.6% of credit RWA).
* **Validation**: 18 grades green, 1 amber, 2 not tested. AUC is between 0.86 and 0.90. The demo is well calibrated by construction.

## 4. Project structure

```
credit-risk-suite/
├── app.py                 # Router (st.navigation) – one page per credit module
├── views/
│   └── ifrs9_page.py      # IFRS 9 page
├── ecl/
│   ├── params.py          # Scales, TTC calibration, products, LGD, scenarios, SICR, IRB floors
│   ├── pd.py              # Z from macro, Vasicek conditioning, lifetime PD term structures
│   ├── engine.py          # EAD / LGD profiles, ECL by scenario, staging, overlays, summaries
│   ├── movements.py       # IFRS 7 reconciliations and transfer matrix
│   ├── irb.py             # IRB risk weights, EL, provision shortfall / excess
│   ├── validation.py      # Binomial and Jeffreys tests, AUC, PSI
│   ├── io.py              # CSV import and validation, scenario tables, settings serialisation
│   └── demo.py            # Synthetic two-date loan book and snapshots
├── templates/             # Demo portfolio (two-date format) and closing snapshot template
├── tests/                 # 23 pytest tests (hand-checked ECL, Basel reference risk weights, reconciliation)
└── .github/workflows/     # CI on Python 3.11 and 3.12
```

## 5. Run locally

```bash
git clone https://github.com/ldenarbonne/credit-risk-suite.git
cd credit-risk-suite
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q
streamlit run app.py
```

## 6. Deploy on Streamlit Community Cloud

1. Push the repository to GitHub (public).
2. On [share.streamlit.io](https://share.streamlit.io), click **Create app**, choose the repository, branch `main` and main file `app.py`. In *Advanced settings*, select Python 3.12.
3. Paste the app URL at the top of this README.

## 7. Limitations

* **Data and model scope**:
  * synthetic data and illustrative calibration; the PD backtest is favourable by construction;
  * one systematic factor per segment and no obligor-level PD model.
* **Accounting scope**:
  * no POCI assets and no modification gains or losses;
  * no interest revenue on the net carrying amount in Stage 3;
  * ECL on undrawn commitments is included in the loss allowance.
* **Sensitivities and IFRS 7**: single-scenario sensitivities keep staging constant. The IFRS 7 presentation is one common convention among several used in practice.
* **IRB**: simplified collateral treatment. The F-IRB secured LGDs (20% property, 25% other physical, 40% haircut) should be checked against CRE32 before any real use.
* **Use**: educational tool; it does not replace an accounting or regulatory calculation.

## 8. References

* IFRS 9 *Financial Instruments*, section 5.5 and Appendix B (B5.5.1–B5.5.55); IFRS 7 *Financial Instruments: Disclosures*, paragraphs 35A–35N.
* EBA/GL/2016/07 on the application of the definition of default; EBA/GL/2017/06 on credit institutions' credit risk management practices and accounting for expected credit losses.
* Basel Framework CRE31 (IRB risk weight functions), CRE32 (risk components), CRE35 (expected losses and provisions); Regulation (EU) No 575/2013, Articles 158–159.
* ECB Banking Supervision (2024), *IFRS 9 overlays and model improvements for novel risks*.

---

**Author:** [Laurent Motais de Narbonne](https://www.linkedin.com/in/laurent-motais-de-narbonne/), Senior Risk Manager. MIT licence.
