# Immuno Compass: Data-Driven Prediction and Analysis of Immunogenicity in Therapeutic Biologics

*Foundations of Data Science course project · final report*

---

## Abstract

Biologic drugs can trigger **anti-drug antibodies (ADAs)**, which may reduce efficacy and affect safety. Using the
**Immunogenicity Database Collaborative (IDC)** we built a cohort-level dataset of **1,754 post-treatment patient groups**
(137 drug products, 108 molecules, 211,018 patients) and modelled the number of ADA-positive patients as
**y ~ Binomial(n, p)**. We compared a Binomial GLM (plain and penalised), XGBoost, and Bayesian Binomial and
**Beta-Binomial** regression under **5-fold cross-validation grouped by molecule**, and tested sequence-derived features
with a pre-registered go/no-go rule. The data are strongly **overdispersed** (Beta-Binomial φ ≈ 3.6; a 1,000-patient cohort
carries the information of ≈ 5 independent patients). For **unseen molecules**, no model clearly outperformed predicting the
database average (best: XGBoost, weighted MAE 12.8 vs 13.1 percentage points, within fold noise). Only Beta-Binomial models
gave honest uncertainty (≈ 88–95 % coverage of 90 % predictive intervals vs ≈ 50 % for Binomial-based models).
Three methods agreed on a short list of associations: **later trial start year** (more sensitive assays), **protein
foreignness** (bacterial and chimeric origin), **active-controlled/biosimilar trials** (all higher ADA), and **higher dose**,
**co-medication** and **healthy-volunteer studies** (lower ADA). Simple whole-molecule sequence features did not give a
consistent improvement. The contribution is an integrated, leakage-aware statistical + Bayesian + ML analysis of the newly
consolidated IDC resource, not a claim that immunogenicity prediction is new.

## 1. Introduction

Therapeutic proteins such as monoclonal antibodies can be recognised as foreign. The resulting ADAs can neutralise the drug,
speed its clearance, or cause hypersensitivity. Immunogenicity is assessed in clinical trials, but results are scattered across
registries, labels and papers. The IDC recently consolidated them. We ask:

1. Can drug, treatment and study information **predict the frequency of ADA-positive patients** for a new molecule?
2. **Which factors are associated** with ADA frequency, and do different modelling approaches agree?

## 2. Data

**Source.** IDC tables: clinical trials (3,334 rows), therapeutics (218), sequences (222 chains), aggregated values and controlled vocabulary.

**Target.** For each row: *n* = patients analysed for ADA, *y* = ADA-positive patients.

**Cleaning decisions** (notebook 02):

| Step | Rows | Rationale |
|---|---|---|
| Raw rows | 3,334 | |
| Valid counts (n > 0, y ≤ n) | 3,281 | 53 rows lacked counts; 7 back-calculated counts rounded and flagged |
| Post-treatment only | 2,653 | "Therapeutic Naive" rows are pre-dose baselines (pooled ADA 4 % vs 15.5 %) |
| One row per cohort | **1,754** | same patients measured at up to 36 timepoints; peak-ADA timepoint kept (latest timepoint used as sensitivity analysis) |

**Features** (16): route, disease area, protein origin ("species"), modality, antibody backbone, expression system, biosimilar
status; dose (parsed from free text to approximate mg/kg; 91 % coverage), trial start year, ADA follow-up length,
co-medication and immunosuppressant flags, placebo mention; blinding, randomisation and comparator. Rare categories
(< 20 cohorts) were grouped. Median imputation, missing-indicators and standardisation were fitted **inside each training fold**.
Drug identity was deliberately **not** a feature.

## 3. Methods

**Evaluation.** 5-fold cross-validation **grouped by molecule** (biosimilars kept with their originator). Metrics:
patient-weighted MAE and RMSE of the ADA frequency, held-out Binomial log-likelihood, calibration, and coverage of 90 %
predictive intervals. Baselines: training-pooled average ("naive") and an intercept-only Beta-Binomial.

**Models.**
* **Binomial GLM** (logit link), with naive, quasi-binomial and cluster-robust (by molecule) standard errors; L1/L2 penalties tuned by nested grouped CV.
* **XGBoost** (`reg:logistic` on y/n, weighted by n or √n), plan settings and a nested-CV tuned variant.
* **Bayesian Binomial** and **Beta-Binomial** logistic regression in PyMC (NUTS, 4 × 1,000 draws). Priors α ~ N(0, 1.5),
  β ~ N(0, 1), φ ~ Exp(0.1). Diagnostics: r̂, ESS, divergences, prior and posterior predictive checks.
* **Sequence features** (Biopython, whole molecule): log length, pI, GRAVY, aromaticity, instability index; pre-registered rule:
  adopt only if better on average **and** in ≥ 4/5 folds.
* **Interpretation:** odds ratios (GLM), posterior odds ratios (Beta-Binomial), SHAP (XGBoost), and a cross-method agreement table.

## 4. Results

### 4.1 Overdispersion
The GLM dispersion was ≈ 22 and 31 % of cohorts fell outside a 99.8 % Binomial funnel (≈ 0.2 % expected). The Beta-Binomial
estimated φ ≈ 3.6 (ρ ≈ 0.22) and passed posterior predictive checks that the Binomial failed (share of 0 % cohorts:
observed 34.7 %, Beta-Binomial 35–40 %, Binomial 21–24 %).

### 4.2 Prediction for unseen molecules

| Model | wMAE | wRMSE | log-lik | 90 % coverage |
|---|---|---|---|---|
| Naive baseline | 0.131 ± 0.037 | 0.183 ± 0.060 | −17.8 | 49 % |
| GLM (plain) | 0.161 ± 0.033 | 0.230 ± 0.041 | −33.4 | – |
| GLM + L2 (nested CV) | 0.132 ± 0.039 | 0.183 ± 0.062 | −18.7 | 50 % |
| **XGBoost (plan settings)** | **0.128 ± 0.038** | **0.180 ± 0.060** | −19.8 | 53 % |
| Bayesian Binomial | 0.160 ± 0.033 | 0.226 ± 0.043 | −29.3 | 47 % |
| Bayesian Beta-Binomial | 0.159 ± 0.034 | 0.209 ± 0.044 | −21.6 | **88 %** |
| Intercept-only Beta-Binomial | – | – | lps −2.82 | **95 %** |

XGBoost's gain over the naive baseline (0.3 points, 4/5 folds) is smaller than its fold SD and absent on log-likelihood.
Models that treat patients as independent over-fit to the largest trials. Calibration plots show predictions spread wider than
observed rates (over-confidence about between-molecule differences). A random (leaky) split roughly halved apparent errors
(e.g. XGBoost wMAE 0.070), which shows the importance of grouped evaluation.

### 4.3 Sequence features
Coverage was 89 % of cohorts (76 distinct sequences). No model met the pre-registered rule on the full data (XGBoost wMAE
0.128 → 0.115 but only 3/5 folds; no log-likelihood gain). On the matched subset, XGBoost improved in 4/5 folds, which is a
hypothesis for future work. **Decision: not adopted.** Notably, grouping folds by product instead of molecule had inflated the
apparent sequence benefit (biosimilars share sequences); this leakage was found and corrected.

### 4.4 Associations (all three methods agree in direction on 32/42 features)

| Association | GLM OR (cluster-robust 95 % CI) | Beta-Binomial OR (94 % CrI) |
|---|---|---|
| Later trial start (per year / per SD) | 1.15 / yr [1.07–1.24] | 1.22 / SD [1.14–1.31] |
| Higher dose (per log unit / per SD) | 0.73 [0.54–0.99] | 0.75 [0.70–0.81] |
| Bacterial vs humanized origin | 45 [1.3–1560] | 6.0 [2.5–12.8] |
| Active-controlled trial | 2.6 [1.3–5.2] | 2.3 [1.8–2.8] |
| Biosimilar | 1.93 [1.20–3.10] | 1.11 / SD [1.05–1.17] |
| Any co-medication | 0.58 [0.43–0.80] | 0.91 / SD [0.84–0.98] |
| Healthy volunteers vs cancer | 0.31 [0.11–0.88] | 0.52 [0.32–0.79] |
| Chimeric vs humanized | 1.9 [0.7–5.3] | 2.0 [1.5–2.5] |
| Subcutaneous vs IV | 1.37 [0.8–2.3] | 1.40 [1.14–1.71] |

The first seven are confident in both models; chimeric origin and subcutaneous route are supported by the Beta-Binomial and
consistent in direction everywhere, but not confirmed by the conservative cluster-robust GLM.

## 5. Discussion

* **Overdispersion is the defining feature of these data.** Ignoring it produces over-confident inference (40/42 "significant"
  effects) and poor predictions. The Beta-Binomial is the appropriate likelihood.
* **Prediction for new molecules is close to the base rate.** The available label-level features capture only a small part of the
  molecule-specific variation, and only ~86 molecules are available for training in each fold.
* **Consistent associations** align with immunology (foreignness, tolerance at high dose, immune status of patients) and with
  measurement (assay sensitivity increasing over time; head-to-head biosimilar trials using sensitive assays). Several are therefore
  likely to reflect **how ADA is measured** rather than biology alone.

## 6. Limitations

Observational, heterogeneous trials and ADA assays; dose parsed approximately from free text; the peak-timepoint target is slightly
optimistic (latest-timepoint sensitivity analysis gave the same conclusions); no per-molecule random effect in the Bayesian model
(conservative cluster-robust GLM used as a check); associations only; ~108 molecules limit statistical power.

## 7. Future work

Per-molecule random effects; assay-type and assay-era covariates; variable-region and humanness sequence features; T-cell
epitope predictions; external validation on newly added IDC trials.

## 8. Reproducibility

All results regenerate from `data/raw/` by running `notebooks/01`–`12` in order (≈ 35 min, mostly Bayesian sampling).
Shared code: `src/immuno.py`. Saved models: `models/`. Dashboard: `app/app.py`.

## Deliverables

1. Re-runnable notebook pipeline: `notebooks/01`–`12`
2. Sequence-feature experiment (tested, not adopted): `notebooks/09_sequence_features.ipynb`
3. Interactive Streamlit dashboard: `app/`
4. Presentation: `slides/`
