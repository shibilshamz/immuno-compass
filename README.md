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

All are compared using **grouped cross-validation by drug**, so that no drug appears in both training and testing.

## Repo layout
```
data/raw/        original IDC CSV files (never edited)
data/processed/  clean tables made by the notebooks
notebooks/    Colab notebooks, numbered in order
app/          Streamlit dashboard
figures/      saved charts
models/       saved model files used by the dashboard
slides/       final presentation
src/          shared code: feature lists, preprocessing, metrics (used by every model notebook)
```

## Progress
- [x] Day 1: load and inspect (`notebooks/01_load_and_inspect.ipynb`)
- [x] Day 2: target definition and cleaning (`notebooks/02_target_and_cleaning.ipynb` → `data/processed/cohorts.csv`, 1,754 cohorts × 137 drugs)
- [x] Day 3: exploratory analysis (`notebooks/03_exploratory_analysis.ipynb`, charts in `figures/`)
- [x] Day 4: features, preprocessing and grouped CV folds (`notebooks/04_features_and_cv.ipynb`, `src/immuno.py`, `data/processed/folds.csv`)
- [ ] Days 5–7: Binomial GLM, XGBoost, checkpoint
- [ ] Days 8–14: Bayesian model, sequence features, comparison, interpretation
- [ ] Days 15–21: figures, dashboard, presentation

## Running in Colab
Open a notebook from GitHub in Colab (File → Open notebook → GitHub, tick "Include private repos").
If the data folder isn't found, the notebook will ask you to upload the 5 CSV files.

### Colab with the private repo (notebooks 04 onwards)
The first cell asks for a GitHub token and clones the repo, so data and `src/` are available.
Create one at github.com/settings/tokens → *Fine-grained token* → repository access: `immuno-compass` → *Contents: Read-only*.
