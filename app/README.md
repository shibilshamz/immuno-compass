# Immuno Compass dashboard

**Live app:** https://immuno-compass-6vv9xjcxqs9rzp6dztbn2x.streamlit.app

An interactive Streamlit app with four tabs:

| Tab | What it does |
|---|---|
| 📊 Explore the data | ADA frequency by origin, route, disease, drug type, dose band, trial period… with filters and molecule-bootstrap intervals |
| 🔮 Predict | describe a drug + trial → XGBoost estimate, Bayesian Beta-Binomial expected rate (94 % credible interval) and the predictive range for a trial group of your chosen size |
| 🏁 Model results | cross-validation scoreboard and the key figures |
| ℹ️ About | question, data, methods, findings, limitations |

The app **never retrains**. It loads `data/processed/cohorts.csv`, `models/xgb_model.joblib`,
`models/bayes_betabinomial.joblib` (each with its fitted preprocessing) and `results/`. Predictions therefore use exactly the
training recipe.

## Open it from Google Colab (easiest, no extra accounts)
Open `notebooks/13_open_dashboard_in_colab.ipynb` in Colab and run all cells. The last cell gives a link that opens the
dashboard in a new tab (live while the Colab session runs).

## Run it on a laptop
```bash
pip install -r app/requirements.txt
streamlit run app/app.py          # from the repository root
```
It opens at http://localhost:8501.

## Put it online for free (Streamlit Community Cloud)
1. Go to **share.streamlit.io** and sign in with the GitHub account that owns `immuno-compass`
   (allow Streamlit to access private repositories when asked).
2. **Create app → Deploy from GitHub**, then choose:
   * Repository: `shibilshamz/immuno-compass`, Branch: `main`
   * Main file path: `app/app.py`
   * **Advanced settings → Python version: 3.11** (the models were saved with Python 3.11 libraries)
3. Click **Deploy**. Streamlit installs `app/requirements.txt` (the file next to `app.py`) automatically.

If the deploy log shows a version error when loading the models, check that `scikit-learn` and `xgboost` in
`app/requirements.txt` match the versions printed by the notebooks that saved the models (06 and 08).
