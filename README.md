# Immuno Compass

**Data-driven prediction and analysis of immunogenicity in therapeutic biologics.**
Foundations of Data Science course project.

## The question
Biologic drugs (such as therapeutic antibodies) are sometimes treated as "foreign" by a patient's immune system,
which then makes **anti-drug antibodies (ADAs)**. Using information about the drug, the treatment and the
clinical study, can we predict **what fraction of patients develop ADAs**, and which factors are most associated with it?

Each data row gives *n* patients tested and *y* ADA-positive patients, so we model **y ~ Binomial(n, p)**.

## Data
Immunogenicity Database Collaborative (IDC), immunogenicitydb.org. Files are in `data/raw/`:

| File | What it is |
|---|---|
| `clinical_trials.csv` | 3,334 rows: one patient group × timepoint per row, with n and y |
| `aggregated_ada_values.csv` | Same rows plus pre-computed cohort / trial / drug-level summaries |
| `therapeutics.csv` | 218 drugs: modality, species, backbone, expression system… |
| `sequences.csv` | 222 amino-acid chains for 120 drugs |
| `controlled_language.csv` | Allowed values for categorical columns |

Linking key: `Therapeutic Assessed for ADA ID` (trials) = `Therapeutic ID` (therapeutics) = `Parental Molecule Therapeutic ID` (sequences).

## Models
1. Binomial GLM (statistical baseline)
2. Bayesian Binomial / Beta-Binomial regression (PyMC)
3. XGBoost

All are compared using **grouped cross-validation by molecule**, so that no molecule (including its biosimilars) appears in both training and testing.

## Repo layout
```
data/raw/        original IDC CSV files (never edited)
data/processed/  clean tables made by the notebooks
notebooks/    Colab notebooks, numbered in order
app/          Streamlit dashboard
figures/      saved charts
models/       saved model files used by the dashboard
reports/      progress summaries
slides/       final presentation
results/      cross-validation scores and out-of-fold predictions per model
src/          shared code: feature lists, preprocessing, metrics (used by every model notebook)
```

## Progress
- [x] Day 1: load and inspect (`notebooks/01_load_and_inspect.ipynb`)
- [x] Day 2: target definition and cleaning (`notebooks/02_target_and_cleaning.ipynb` → `data/processed/cohorts.csv`, 1,754 cohorts × 137 drugs)
- [x] Day 3: exploratory analysis (`notebooks/03_exploratory_analysis.ipynb`, charts in `figures/`)
- [x] Day 4: features, preprocessing and grouped CV folds (`notebooks/04_features_and_cv.ipynb`, `src/immuno.py`, `data/processed/folds.csv`)
- [x] Day 5: Binomial GLM, overdispersion, regularisation, sensitivity (`notebooks/05_binomial_glm.ipynb`; CV results in `results/`)
- [x] Day 6: XGBoost with nested tuning, first model scoreboard, saved model (`notebooks/06_xgboost.ipynb`, `models/xgb_model.joblib`)
- [x] Day 7: checkpoint, full re-run from raw data verified identical; Week 1 summary (`reports/week1_summary.md`)
- [x] Days 8–9: Bayesian Binomial (`07_bayesian_binomial`)
- [x] Day 10: Bayesian Beta-Binomial + posterior predictive check (`08_beta_binomial`, `models/bayes_betabinomial.joblib`)
- [x] Days 11–12: sequence features, tested, no consistent improvement (`09_sequence_features`); CV regrouped by molecule
- [x] Day 13: full model comparison, calibration, coverage (`10_model_comparison`)
- [x] Day 14: interpretation, SHAP, cross-method agreement (`11_interpretation`); Week 2 summary (`reports/week2_summary.md`)
- [x] Day 15: final slide figures (`12_final_figures`, `figures/final/`)
- [x] Days 16–17: Streamlit dashboard (`app/app.py`, see `app/README.md`; open it from Colab with `notebooks/13_open_dashboard_in_colab.ipynb`)
- [x] Day 18: final report (`reports/final_report.md`); fresh-clone re-run of notebooks 01–12 reproduced all 27 data/result files exactly
- [x] Days 19–21: presentation (`slides/immuno_compass.pptx`), speaker script and Q&A preparation (`slides/`)

## Running in Colab
Open a notebook from GitHub in Colab (File → Open notebook → GitHub, tick "Include private repos").
If the data folder isn't found, the notebook will ask you to upload the 5 CSV files.

### Colab (notebooks 04 onwards)
The first cell downloads the repository automatically. If the repository is private, it asks for a GitHub token instead:
create one at github.com/settings/tokens → *Fine-grained token* → repository access: `immuno-compass` → *Contents: Read-only*.
