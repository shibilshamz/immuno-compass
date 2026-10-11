# Immuno Compass · speaker script

About **20–25 minutes** for 18 slides (≈ 1–1.5 min each), leaving time for questions. The same text is in the deck's speaker notes.
Tip: practise out loud twice with a timer; slides 10–14 (results) deserve the most time.

## Slide 1 · Immuno Compass

Hello everyone. My project is called Immuno Compass. It is about a safety and effectiveness problem with modern biologic medicines: sometimes the patient's immune system attacks the drug. I used a newly consolidated public database of clinical trials to ask two questions: can we predict how often this happens for a drug, and which factors are linked to it? I will walk you through the data, the three modelling approaches, an honest comparison, and what the models agree on.

## Slide 2 · When the immune system attacks the medicine

Biologics are protein drugs, for example monoclonal antibodies. Because they are proteins, the immune system can sometimes see them as foreign and produce anti-drug antibodies, which I will call ADAs. ADAs can block the drug, remove it from the body faster, or cause reactions. Every new biologic is tested for ADAs in clinical trials, but the results were scattered. The Immunogenicity Database Collaborative recently collected them. My question: using what we know about the drug, the treatment and the trial, can we predict the ADA frequency, and what is it linked to? I am not claiming immunogenicity prediction is new; the contribution is an integrated statistical, Bayesian and machine-learning analysis of this new resource.

## Slide 3 · Each data row is a count: y of n patients

Here is what one row of data means. A trial group of, say, 100 patients was tested, and 15 were ADA-positive: a frequency of 15 percent. Because we know both counts, this is neither a yes/no classification nor ordinary regression on a percentage. The natural model is the Binomial distribution: each patient is like a coin flip with probability p of being ADA-positive. Our models try to learn how p depends on the drug and the trial. An important consequence: a group of 1,000 patients carries more information than a group of 10, so groups must be weighted.

## Slide 4 · From 3,334 raw rows to 1,754 patient groups

The clinical trials table has 3,334 rows. I removed 53 rows without usable counts. Then an important discovery: rows labelled 'therapeutic naive' are measurements taken before the drug was given, with only 4 percent ADA, so they can't tell us about the drug and were removed. Next, the same patients were often measured at many timepoints, up to 36 rows for one group; counting them all would count the same people many times, so I kept one row per group, at the peak ADA timepoint, and checked that using the last timepoint gives the same conclusions. The result: 1,754 patient groups, 137 drug products from 108 molecules, and over 211,000 patients.

## Slide 5 · The more 'foreign' the protein, the more ADA

This chart already tells a biological story. Fully human antibodies have about 10 percent ADA, humanized ones 13, chimeric antibodies, which are partly mouse, 25 percent, and drugs derived from bacteria almost 90 percent. The less human the protein looks, the more the immune system reacts. The horizontal lines are 95 percent intervals. I computed them by re-sampling molecules rather than individual trial groups, because many groups belong to the same molecule and are not independent. That idea of grouping by molecule comes back several times.

## Slide 6 · Patient groups vary far more than chance allows

If all groups shared one ADA probability, results would scatter only by chance, inside this orange funnel: wide for small groups, narrow for big ones. In reality 31 percent of groups fall outside, where we would expect only 0.2 percent. This is called overdispersion. In the Bayesian Beta-Binomial model, the dispersion parameter phi is about 3.6, which means patients in the same group are correlated: a 1,000-patient trial carries roughly as much information about the average rate as five independent patients. This one fact explains most of what follows.

## Slide 7 · A fair test: predict unseen molecules

How do we test fairly? If we split trial groups randomly, the same drug appears in both training and test data. This bar chart shows a 'drug memoriser' that knows no biology and only remembers each drug's past ADA rate. With a random split it looks brilliant, 7.5 points of error. With an honest split it is no better than guessing the average. So all models are evaluated on molecules they have never seen. One lesson from the project: I first grouped by drug product, but biosimilars are copies of the same molecule. When the sequence features exposed this, I regrouped by molecule and re-ran everything.

## Slide 8 · Three models, one question

I compared three families of models. First, the Binomial GLM, a classic weighing scale where each clue gets a weight; it gives odds ratios. I used cluster-robust errors and tried L1 and L2 regularisation, choosing the penalty with nested cross-validation. Second, the Bayesian model in PyMC. Instead of one number per effect it gives a whole distribution, like a weather forecaster saying 70 percent chance of rain. The Beta-Binomial version lets every trial group have its own probability, which handles the overdispersion. Third, XGBoost, which builds hundreds of small decision trees, like playing twenty questions. All were scored the same way, against a simple baseline: always predict the training average.

## Slide 9 · The Bayesian model in one picture

Bayes' theorem in one picture: the posterior is proportional to the likelihood times the prior. The priors say effects are probably modest. The likelihood is Beta-Binomial: each group first gets its own probability drawn around the model's prediction, then its patients are counted. PyMC samples the posterior with NUTS; the diagnostics were clean, with r-hat of 1.00 and no divergences. Why Beta-Binomial? A plain Binomial version assumes every patient in a group flips the same coin. Its uncertainty ranges were far too narrow. The Beta-Binomial passes the posterior predictive check: simulated data look like the real data, including the share of groups with zero ADA.

## Slide 10 · For a new molecule, no model beats the average

This is the central and honest result. Bars show the average prediction error on molecules never seen in training, with spread across the five folds. Guessing the average gives 13.1 percentage points. XGBoost is best at 12.8, but that small gain is less than the variation between folds, and it does not improve the probability score. Models that treat every patient as independent, the plain GLM and the Bayesian Binomial, do worse than the average. So, with these clues, ADA for a brand-new molecule is barely predictable. Most of the variation is specific to each molecule.

## Slide 11 · Only the Beta-Binomial gives honest uncertainty

Prediction is not only about the middle guess; it is also about honest uncertainty. A 90 percent range should contain the real result about 90 percent of the time. Every model that assumes plain Binomial spread manages only about half. The Beta-Binomial reaches 88 percent. Interestingly, a Beta-Binomial with no clues at all does slightly better still. So the most honest forecast for a new molecule is: around 13 to 14 percent on average, with a wide range. The take-home message: getting the noise model right mattered more than any clue.

## Slide 12 · Predictions are too spread out

Calibration asks: when the model says 20 percent, is it 20 percent? I sorted held-out groups by their predicted ADA into eight bins and compared the average prediction with what was observed. A perfect model sits on the diagonal. Our models predict a range from about 2 to 35 percent, but the observed rates in those bins stay between about 4 and 26 percent, closer to the middle. The models are over-confident about how different a new molecule will be. That is consistent with everything else: limited information about each molecule, and a lot of noise.

## Slide 13 · What is linked to ADA (associations, not causes)

Even if prediction for new molecules is hard, the models can still describe patterns. These are odds ratios from the Bayesian Beta-Binomial with 94 percent credible intervals. Orange means more ADA, blue less. More foreign proteins, bacterial or chimeric, have more ADA, and so does injection under the skin compared with into a vein. Later trials report more ADA, which is most likely because newer tests are more sensitive. Higher doses and co-medication go with less ADA, consistent with immune tolerance and suppression. These are associations in observational data, not causes.

## Slide 14 · Three methods agree on the core factors

To check these patterns, I compared three very different explanations: GLM odds ratios with cluster-robust errors, Bayesian posteriors, and SHAP values from XGBoost. On the left, the SHAP plot: each dot is a trial group, and its position shows how much a clue pushed that group's prediction up or down. Across all 42 feature columns, 32 point in the same direction in all three methods, and 9 are confidently non-zero in both statistical models. That agreement gives me more confidence that these are real patterns in the database, while still being associations.

## Slide 15 · Protein-sequence features: tested, not adopted

I also promised to try molecular features. From the sequence table I computed five whole-molecule properties with Biopython: size, charge balance, hydrophobicity, aromaticity and an instability index, for 89 percent of groups. Before looking at results I fixed a rule: keep them only if they improve the error on average and in at least four of the five folds. No model passed. XGBoost improved on average but in only three folds. So the honest report is: tested, no consistent improvement. The experiment also revealed the biosimilar leak I mentioned, which made the first, product-grouped result look much better than it really was.

## Slide 16 · Interactive dashboard

The fourth deliverable is an interactive Streamlit dashboard. In the Explore tab you can group ADA frequency by origin, route, disease area or dose and filter the data. In the Predict tab you describe a drug and a trial and get the XGBoost estimate and the Bayesian Beta-Binomial range, including how many patients out of a group of your chosen size might be ADA-positive. The app never retrains; it loads the saved models together with their preprocessing, which avoids the classic mismatch between training and the app. It is online at https://immuno-compass-6vv9xjcxqs9rzp6dztbn2x.streamlit.app, and if time allows I can show it live.

## Slide 17 · Limitations and next steps

Some limitations. The trials used different ADA tests over two decades, which is probably why trial year matters so much. Dose had to be extracted from text and is approximate. With only about 108 molecules there is limited information about drug-specific effects, and the Bayesian model does not yet include a per-molecule random effect; I used cluster-robust GLM errors as a conservative check. Everything is association, not causation. Natural next steps: add assay information, build a hierarchical model, and try more targeted sequence features such as humanness scores and T-cell epitopes.

## Slide 18 · Take-home messages

To summarise. First, these data are strongly overdispersed, and only a Beta-Binomial model gives honest uncertainty. Second, for a brand-new molecule, the clues we have predict ADA barely better than the database average of about 13 percent. Third, despite that, three very different methods agree on a short list of associated factors: assay era, dose, how foreign the protein is, the type of trial, and co-medication. And fourth, a methodological lesson: fair evaluation by molecule changes the conclusions, and leakage is easy to miss. Thank you, I am happy to take questions.
