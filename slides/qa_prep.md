# Immuno Compass · Q&A preparation

Likely questions from the professor or classmates, with short honest answers and where the evidence is.

## Data and target

**Why not treat ADA as yes/no classification, or the % as ordinary regression?**
Each row has two counts: *n* tested and *y* positive. The Binomial uses both, so a 1,000-patient group weighs more than a 10-patient group, and predictions stay between 0 and 1. (notebook 02, slide 3)

**Why did you drop the "Therapeutic Naive" rows?**
They are measurements before dosing (pooled ADA 4 % vs 15.5 % after treatment), so they cannot tell us about the drug's effect. (notebook 02)

**The same patients appear at several timepoints. How did you handle that?**
One row per patient group, at its peak ADA timepoint. A "latest timepoint" version gave the same odds ratios (e.g. start year 1.15 vs 1.16 per year). (notebooks 02, 05)

**How reliable is the dose variable?**
Approximate: parsed from free text and converted to mg/kg assuming a 70 kg adult and 1.8 m² body surface. 91 % coverage. We say so explicitly. (notebook 02)

## Methods

**Why grouped cross-validation by molecule?**
Groups of the same molecule are correlated. A random split lets a model "recognise" drugs: a memoriser drops from 13.1 to 7.5 points of error under a random split. Biosimilars are the same molecule, so they stay together. (notebook 04, slide 7)

**What is overdispersion, and how do you know it's there?**
More variation than the Binomial allows: dispersion ≈ 22 in the GLM, 31 % of groups outside a 99.8 % chance funnel, Beta-Binomial φ ≈ 3.6. (notebooks 03, 05, 08)

**Explain the Beta-Binomial in one sentence.**
Each trial group gets its own ADA probability drawn from a Beta distribution centred on the model's prediction; then its patients are counted Binomially.

**Why these priors?**
Weakly informative: β ~ N(0, 1) says an odds ratio beyond ~7 per SD is surprising; α ~ N(0, 1.5); φ ~ Exp(0.1) allows anything from near-Binomial to very overdispersed. The prior predictive check covers 0–100 %, and 211,000 patients dominate. (notebook 07)

**How do you know the sampler worked?**
r̂ = 1.00, bulk ESS > 2,500, zero divergences, in the full fit and every CV fold; trace plots mix well. (notebooks 07, 08)

**Why is a credible interval different from a confidence interval?**
A 94 % credible interval means: given the model and data, there is a 94 % probability the effect lies inside. A 95 % confidence interval is a statement about the procedure over repeated samples.

**Did you tune hyper-parameters on the test data?**
No. Penalties and XGBoost settings were chosen by an *inner* grouped CV inside each training fold (nested CV). We also show the "best in hindsight" setting only as an optimistic ceiling. (notebooks 05, 06)

## Results

**So the models don't work?**
They don't predict ADA for a **new molecule** much better than the database average, which is an honest and useful finding: most variation is molecule-specific and not captured by label-level clues. The models still reveal consistent **associations**, and the Beta-Binomial gives honest uncertainty. (notebook 10)

**Is XGBoost's win significant?**
Its gain (0.3 percentage points, better in 4/5 folds) is smaller than the fold-to-fold SD, and it is worse on log-likelihood. We don't claim it is better. (notebook 10)

**Why does the plain GLM do worse than the average?**
It treats every patient as independent (so giant trials dominate), has 42 weights for ~86 training molecules, and over-fits. Regularisation brings it back to baseline level. (notebook 05)

**Why does trial start year matter so much?**
Most likely assay sensitivity: newer, drug-tolerant ADA tests detect more. It's a measurement effect, not evidence that newer drugs are worse.

**Isn't the immunosuppressant result surprising?**
The raw comparison is confounded by disease area (immunosuppressants are used in autoimmune and cancer trials). After adjustment there is no consistent effect; "any co-medication" is associated with lower ADA. (notebooks 03, 11)

**Do the methods agree?**
32 of 42 features point the same way in GLM, Bayesian model and SHAP; 9 are confident in both statistical models. (notebook 11, slide 14)

## Sequence features

**Why didn't sequence features help?**
Most drugs are IgG antibodies with near-identical whole-molecule properties (constant regions dominate), there are only ~76 distinct sequences, and immunogenicity depends on specific regions (variable domains, T-cell epitopes). A pre-registered rule decided: not adopted. (notebook 09)

**You changed the grouping halfway. Isn't that suspicious?**
It was a correction for leakage that we found ourselves (biosimilars share sequences), and it made results *less* flattering. Every affected notebook was re-run and every number updated. (Week 2 summary)

## Dashboard

**Could the dashboard give different predictions from the notebook?**
No. It loads the saved models together with their fitted preprocessing; nothing is re-implemented. It was tested headlessly (Streamlit AppTest) and in a browser. (app/)

## Big picture

**What is new here?**
Not immunogenicity prediction itself, but a leakage-aware, integrated statistical + Bayesian + ML comparison on the newly consolidated IDC resource, with honest uncertainty and a pre-registered test of sequence features.

**What would you do with more time?**
Add assay type as a covariate, a hierarchical model with per-molecule random effects, humanness and T-cell epitope features, and validate on new IDC trials.
