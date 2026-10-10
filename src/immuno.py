"""Shared helpers for the Immuno Compass notebooks.

Every model notebook imports from here, so that all three models see exactly the
same features, the same preprocessing and the same train/test folds, and are
scored with the same metrics. Keeping this in one place is what makes the model
comparison on Day 13 fair.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DRUG = "Therapeutic Assessed for ADA ID"   # the grouping key: one id per drug

# ---------------------------------------------------------------------------
# Features (the reason for each one is documented in notebook 04)
# ---------------------------------------------------------------------------
CAT_FEATURES = [
    "route",        # how the drug enters the body
    "indication",   # disease area
    "blinding",     # study design
    "randomized",   # study design
    "comparator",   # study design
    "modality",     # kind of drug
    "species",      # origin of the protein design (human ... bacterial)
    "backbone",     # antibody skeleton class
    "expression",   # cells used to make the drug
]
NUM_FEATURES = [
    "log_dose_mgkg",      # dose, log scale
    "start_year",         # proxy for ADA assay generation
    "log_followup_days",  # how long ADAs were looked for
]
FLAG_FEATURES = [
    "co_immunosuppressant",  # given with immune-calming drugs
    "co_any_drug",           # given with any other drug
    "arm_mentions_placebo",  # arm description mentions placebo
    "biosimilar",            # drug is a biosimilar copy
]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES + FLAG_FEATURES

# Reference level for each category: the most common, "typical" value. Odds ratios and
# coefficients are read as "compared with this level".
REFERENCE = {
    "route": "Intravenous",
    "indication": "Cancer and neoplasms",
    "blinding": "Double",
    "randomized": "Randomized",
    "comparator": "Placebo/Sham",
    "modality": "Monoclonal antibody",
    "species": "Humanized",
    "backbone": "Human IgG1",
    "expression": "Chinese hamster ovary (CHO) cells",
}


def find_repo_root(start=None):
    """Walk upwards until we find the folder that contains data/processed/cohorts.csv."""
    here = Path(start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "processed" / "cohorts.csv").exists():
            return p
    raise FileNotFoundError("Could not find data/processed/cohorts.csv - run notebook 02 first.")


def load_cohorts(root=None, which="peak"):
    """Load the clean one-row-per-cohort table and apply the final feature tweaks."""
    root = Path(root) if root else find_repo_root()
    name = "cohorts.csv" if which == "peak" else "cohorts_latest_timepoint.csv"
    df = pd.read_csv(root / "data" / "processed" / name)
    # 'Unspecified' disease area is almost identical to ophthalmic route (r = 0.88): fold it into 'Other'.
    df["indication"] = df["indication"].replace({"Unspecified": "Other"})
    return df


def load_folds(root=None):
    root = Path(root) if root else find_repo_root()
    return pd.read_csv(root / "data" / "processed" / "folds.csv")


def make_preprocessor(scale=True):
    """Turn the raw feature columns into a numeric matrix.

    * categoricals -> one-hot, dropping the REFERENCE level; unseen levels in a test fold -> all zeros
    * numerics     -> median imputation + a 0/1 'was missing' column, then (optionally) standardised
    * 0/1 flags    -> kept as they are (standardised too when scale=True, for the GLM / Bayesian prior)

    It must be FIT ON THE TRAINING FOLD ONLY, so the test fold never influences medians or scaling.
    """
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    flag_steps = [("scale", StandardScaler())] if scale else [("keep", "passthrough")]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop=[REFERENCE[c] for c in CAT_FEATURES], handle_unknown="ignore",
                                  sparse_output=False), CAT_FEATURES),
            ("num", Pipeline(num_steps), NUM_FEATURES),
            ("flag", Pipeline(flag_steps), FLAG_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Metrics: every model is scored with exactly these
# ---------------------------------------------------------------------------
def weighted_mae(y, n, p):
    """Average |predicted - observed| ADA frequency, each cohort weighted by its size n."""
    return np.average(np.abs(p - y / n), weights=n)


def weighted_rmse(y, n, p):
    return np.sqrt(np.average((p - y / n) ** 2, weights=n))


def mean_loglik(y, n, p):
    """Held-out Binomial log-likelihood per cohort (higher = better)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return binom.logpmf(y, n, p).mean()


def score(y, n, p):
    y, n, p = map(np.asarray, (y, n, p))
    return {"wMAE": weighted_mae(y, n, p), "wRMSE": weighted_rmse(y, n, p), "loglik": mean_loglik(y, n, p)}


def cross_validate(df, fold_col, fit_predict):
    """Run fit_predict(train_df, test_df) -> predicted probabilities, on every fold.

    Returns (per-fold score table, out-of-fold predictions aligned with df).
    """
    oof = np.full(len(df), np.nan)
    rows = []
    for k in sorted(df[fold_col].unique()):
        test = df[fold_col] == k
        p = np.asarray(fit_predict(df[~test], df[test]), dtype=float)
        oof[test.values] = p
        rows.append({"fold": k, **score(df.loc[test, "y"], df.loc[test, "n"], p)})
    return pd.DataFrame(rows), oof


# ---------------------------------------------------------------------------
# Model 1: Binomial GLM (optionally penalised) - shared by notebooks 05 and 13
# ---------------------------------------------------------------------------
def fit_glm(train, alpha=0.0, L1_wt=0.0):
    """Fit a Binomial GLM on (y successes, n - y failures). Returns (preprocessor, params).

    alpha = 0       -> plain maximum-likelihood GLM
    alpha > 0       -> penalised: L1_wt = 1 is lasso (L1), L1_wt = 0 is ridge (L2).
    The intercept is never penalised.
    """
    import statsmodels.api as sm
    pre = make_preprocessor(scale=True).fit(train)
    X = sm.add_constant(pre.transform(train), has_constant="add")
    endog = np.column_stack([train["y"], train["n"] - train["y"]])
    model = sm.GLM(endog, X, family=sm.families.Binomial())
    if alpha == 0:
        params = model.fit().params
    else:
        penalty = np.r_[0.0, np.full(X.shape[1] - 1, alpha)]
        params = model.fit_regularized(method="elastic_net", alpha=penalty, L1_wt=L1_wt, maxiter=500).params
    return pre, np.asarray(params)


def predict_glm(pre, params, test):
    X = np.column_stack([np.ones(len(test)), pre.transform(test)])
    return 1 / (1 + np.exp(-(X @ params)))


def save_results(name, per_fold, oof, df, root=None):
    """Store per-fold scores and out-of-fold predictions so notebook 13 can compare all models."""
    root = Path(root) if root else find_repo_root()
    out = root / "results"
    out.mkdir(exist_ok=True)
    per_fold.assign(model=name).to_csv(out / f"cv_{name}.csv", index=False)
    pd.DataFrame({"cohort_id": df["cohort_id"].values, "pred": oof}).to_csv(out / f"oof_{name}.csv", index=False)
