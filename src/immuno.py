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
    "dose_type",    # how the dose was written (per kg / flat / per m2 / unknown)
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

    * categoricals -> one-hot (first level dropped); unseen levels in a test fold -> all zeros
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
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
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
