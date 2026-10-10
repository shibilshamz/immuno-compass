"""Immuno Compass dashboard.

Run locally from the repo root:   streamlit run app/app.py

The app never trains anything. It loads the cleaned data, the saved models (with their fitted preprocessing)
and the saved cross-validation results produced by the notebooks, so predictions use exactly the same
feature recipe as training.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import immuno as im  # noqa: E402  (shared feature lists and data loading)

st.set_page_config(page_title="Immuno Compass", page_icon="🧭", layout="wide")

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"


# ---------------------------------------------------------------------------
# Loading (cached so the app stays fast)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    return im.load_cohorts(ROOT)


@st.cache_resource
def load_models():
    xgb = joblib.load(ROOT / "models" / "xgb_model.joblib")
    bayes = joblib.load(ROOT / "models" / "bayes_betabinomial.joblib")
    return xgb, bayes


@st.cache_data
def load_scoreboard():
    names = {"naive": "Guess the average", "glm_l2": "GLM + L2", "xgb_default": "XGBoost",
             "bayes_betabinomial": "Bayesian Beta-Binomial", "glm_plain": "GLM (plain)",
             "bayes_binomial": "Bayesian Binomial"}
    cv = pd.concat([pd.read_csv(ROOT / "results" / f"cv_{k}.csv") for k in names])
    t = cv.groupby("model")[["wMAE", "wRMSE", "loglik"]].agg(["mean", "std"]).reindex(list(names))
    t.index = [names[k] for k in t.index]
    t.columns = [f"{m} ({s})" for m, s in t.columns]
    return t


@st.cache_data
def pooled_by(data, col):
    """Pooled ADA % per level with a 95 % interval from re-sampling molecules (not cohorts)."""
    rows = []
    rng = np.random.default_rng(42)
    for level, g in data.groupby(col, observed=True):
        per = g.groupby(im.GROUP)[["y", "n"]].sum().values
        lo = hi = np.nan
        if len(per) >= 2:
            idx = rng.integers(0, len(per), size=(1000, len(per)))
            boot = per[idx, 0].sum(1) / per[idx, 1].sum(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({col: str(level), "ADA %": 100 * g["y"].sum() / g["n"].sum(), "low": 100 * lo, "high": 100 * hi,
                     "patient groups": len(g), "molecules": g[im.GROUP].nunique(), "patients": int(g["n"].sum())})
    return pd.DataFrame(rows).sort_values("ADA %")


df = load_data()
xgb_bundle, bayes = load_models()
overall = df["y"].sum() / df["n"].sum()

LABELS = {"species": "Protein origin", "route": "Route", "indication": "Disease area", "modality": "Drug type",
          "backbone": "Antibody backbone", "expression": "Made in (cells)", "blinding": "Blinding",
          "randomized": "Randomised", "comparator": "Comparator", "dose_band": "Dose band", "year_band": "Trial start"}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧭 Immuno Compass")
st.markdown("How often do patients develop **anti-drug antibodies (ADA)** against biologic drugs, and what is it linked to? "
            "Data: Immunogenicity Database Collaborative (IDC) · "
            f"**{len(df):,} patient groups**, **{df[im.DRUG].nunique()} drugs**, **{int(df['n'].sum()):,} patients**.")
st.caption("All results are associations in observational clinical-trial data, not causal effects.")

tab_explore, tab_predict, tab_models, tab_about = st.tabs(["📊 Explore the data", "🔮 Predict", "🏁 Model results", "ℹ️ About"])

# ---------------------------------------------------------------------------
# Tab 1: explore
# ---------------------------------------------------------------------------
with tab_explore:
    data = df.copy()
    data["dose_band"] = pd.cut(data["dose_mgkg"], [0, 0.3, 1, 3, 10, np.inf], include_lowest=True,
                               labels=["< 0.3 mg/kg", "0.3–1", "1–3", "3–10", "> 10"]).astype(str).replace("nan", "Unknown")
    data["year_band"] = pd.cut(data["start_year"], [1990, 2005, 2009, 2012, 2015, 2018, 2025],
                               labels=["≤2005", "2006–09", "2010–12", "2013–15", "2016–18", "2019+"]).astype(str).replace("nan", "Unknown")

    c1, c2, c3 = st.columns([1.2, 1.4, 1.4])
    group_col = c1.selectbox("Group ADA frequency by", list(LABELS), format_func=LABELS.get)
    diseases = c2.multiselect("Disease areas", sorted(data["indication"].unique()), placeholder="All disease areas")
    routes = c3.multiselect("Routes", sorted(data["route"].unique()), placeholder="All routes")
    if diseases:
        data = data[data["indication"].isin(diseases)]
    if routes:
        data = data[data["route"].isin(routes)]

    if data.empty:
        st.warning("No patient groups match these filters.")
    else:
        t = pooled_by(data, group_col)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t["ADA %"], y=t[group_col], mode="markers", marker=dict(size=14, color=BLUE),
            error_x=dict(type="data", symmetric=False, array=(t["high"] - t["ADA %"]).fillna(0),
                         arrayminus=(t["ADA %"] - t["low"]).fillna(0), color=BLUE, thickness=3, width=0),
            customdata=t[["molecules", "patient groups", "patients"]],
            hovertemplate="<b>%{y}</b><br>ADA %{x:.1f}%<br>%{customdata[0]} molecules · %{customdata[1]} groups · "
                          "%{customdata[2]:,} patients<extra></extra>"))
        fig.add_vline(x=100 * overall, line_dash="dash", line_color=GRAY,
                      annotation_text=f"all data {overall:.0%}", annotation_position="top")
        fig.update_layout(height=120 + 45 * len(t), margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                          xaxis_title="% of patients with ADA (pooled, 95 % interval from re-sampling molecules)",
                          yaxis_title=None, xaxis=dict(rangemode="tozero"))
        st.subheader(f"ADA by {LABELS[group_col].lower()}")
        st.plotly_chart(fig, width="stretch")
        st.caption("Wide intervals mean few molecules in that group. Raw comparisons can be confounded "
                   "(e.g. immunosuppressants are mostly used in autoimmune and cancer trials).")
        with st.expander("Show the table"):
            st.dataframe(t.round(1), hide_index=True, width="stretch")

        st.subheader("Every patient group")
        fig2 = px.scatter(data, x="n", y=data["ada_freq"] * 100, log_x=True, opacity=0.5,
                          hover_data={"drug_name": True, "indication": True, "n": True},
                          labels={"n": "patients tested (log scale)", "y": "% ADA-positive"},
                          color_discrete_sequence=[BLUE])
        fig2.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="% ADA-positive")
        st.plotly_chart(fig2, width="stretch")
        st.caption("Small groups scatter widely by chance; big groups still differ far more than chance allows (overdispersion).")

# ---------------------------------------------------------------------------
# Tab 2: predict
# ---------------------------------------------------------------------------
def levels(col):
    """Values seen in training, most common first (so the default is the typical value)."""
    return list(df[col].value_counts().index)


with tab_predict:
    st.markdown("Describe a drug and a trial. You get **two models'** estimates of the ADA frequency. "
                "Both were tested on molecules they had never seen, where neither beat the database average by much, "
                "so read these as *rough, uncertain* estimates.")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown("**The drug**")
        species = st.selectbox("Protein origin", levels("species"))
        modality = st.selectbox("Drug type", levels("modality"))
        backbone = st.selectbox("Antibody backbone", levels("backbone"))
        expression = st.selectbox("Made in (cells)", levels("expression"))
        biosimilar = st.checkbox("Biosimilar (copy of an approved drug)")
        st.markdown("**The treatment**")
        route = st.selectbox("Route", levels("route"))
        dose_known = st.checkbox("Dose known", value=True)
        dose = st.number_input("Dose (mg per kg body weight)", min_value=0.0, max_value=100.0, value=3.0, step=0.5,
                               disabled=not dose_known, help="For a flat dose, divide mg by ~70 kg.")
        co_any = st.checkbox("Given together with other medicines")
        co_imm = st.checkbox("…including immunosuppressants (e.g. methotrexate, steroids)", disabled=not co_any)
        st.markdown("**The trial**")
        indication = st.selectbox("Disease area", levels("indication"))
        start_year = st.slider("Trial start year", 2000, 2021, 2014)
        follow_weeks = st.slider("ADA follow-up (weeks)", 1, 260, 36)
        with st.expander("Trial design (optional)"):
            blinding = st.selectbox("Blinding", levels("blinding"))
            randomized = st.selectbox("Randomised", levels("randomized"))
            comparator = st.selectbox("Comparator", levels("comparator"))
            placebo = st.checkbox("Arm description mentions placebo")
        n_patients = st.slider("Patients in the group (for the expected range)", 10, 1000, 100, step=10)

    row = pd.DataFrame([{
        "route": route, "indication": indication, "blinding": blinding, "randomized": randomized,
        "comparator": comparator, "modality": modality, "species": species, "backbone": backbone,
        "expression": expression,
        "log_dose_mgkg": np.log1p(dose) if dose_known else np.nan,
        "start_year": start_year, "log_followup_days": np.log1p(follow_weeks * 7),
        "co_immunosuppressant": int(co_any and co_imm), "co_any_drug": int(co_any),
        "arm_mentions_placebo": int(placebo), "biosimilar": int(biosimilar),
    }])[im.ALL_FEATURES]

    # XGBoost: point prediction through its own saved preprocessing
    p_xgb = float(xgb_bundle["model"].predict(xgb_bundle["preprocessor"].transform(row))[0])

    # Bayesian Beta-Binomial: 1,000 posterior draws
    x = bayes["preprocessor"].transform(row)[0]
    mu = 1 / (1 + np.exp(-(bayes["alpha"] + bayes["beta"] @ x)))          # expected rate, one per draw
    rng = np.random.default_rng(0)
    p_group = rng.beta(mu * bayes["phi"], (1 - mu) * bayes["phi"])          # this particular group's own rate
    y_sim = rng.binomial(n_patients, p_group)
    mu_lo, mu_med, mu_hi = np.percentile(mu, [3, 50, 97])
    y_lo, y_hi = np.percentile(y_sim, [5, 95])

    with right:
        st.markdown("**Estimates**")
        m1, m2, m3 = st.columns(3)
        m1.metric("XGBoost", f"{p_xgb:.0%}")
        m2.metric("Bayesian (expected)", f"{mu_med:.0%}", help="Posterior median of the expected ADA frequency")
        m3.metric("Database average", f"{overall:.0%}")
        st.markdown(f"- Bayesian **94 % credible interval for the expected rate**: **{mu_lo:.0%} – {mu_hi:.0%}**\n"
                    f"- But single trial groups vary a lot around that (overdispersion). In a group of **{n_patients}** "
                    f"patients, expect **{int(y_lo)} – {int(y_hi)}** with ADA (90 % range, "
                    f"{y_lo / n_patients:.0%} – {y_hi / n_patients:.0%}).")
        hist = np.histogram(y_sim / n_patients * 100, bins=np.linspace(0, 100, 41))
        fig3 = go.Figure(go.Bar(x=hist[1][:-1] + 1.25, y=hist[0] / hist[0].sum() * 100, marker_color=AQUA, width=2.3,
                                hovertemplate="%{x:.0f}% ADA: %{y:.1f}% chance<extra></extra>"))
        fig3.add_vline(x=100 * p_xgb, line_color=BLUE, annotation_text="XGBoost", annotation_position="top")
        fig3.add_vline(x=100 * overall, line_color=GRAY, line_dash="dash", annotation_text="average",
                       annotation_position="bottom right")
        fig3.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                           xaxis_title=f"% of {n_patients} patients with ADA (Bayesian predictive distribution)",
                           yaxis_title="chance (%)")
        st.plotly_chart(fig3, width="stretch")
        st.info("**How to read this.** The bars show how ADA could turn out for one new trial group like this. "
                "They are wide because trials of similar drugs really do differ a lot. This tool is for "
                "exploring patterns in the data, **not** for clinical or development decisions.")

# ---------------------------------------------------------------------------
# Tab 3: model results
# ---------------------------------------------------------------------------
with tab_models:
    st.markdown("All models were scored on **molecules never seen in training** (5-fold cross-validation grouped by molecule; "
                "biosimilars kept with their originator).")
    st.dataframe(load_scoreboard().style.format("{:.3f}"), width="stretch")
    st.caption("wMAE / wRMSE: error in ADA frequency weighted by patients (lower is better). "
               "loglik: held-out Binomial log-likelihood per group (higher is better). Values: mean and SD over 5 folds.")
    fig_dir = ROOT / "figures" / "final"
    a, b = st.columns(2)
    a.image(str(fig_dir / "06_model_comparison.png"), width="stretch")
    b.image(str(fig_dir / "07_coverage.png"), width="stretch")
    a.image(str(fig_dir / "08_calibration.png"), width="stretch")
    b.image(str(fig_dir / "09_what_drives_ada.png"), width="stretch")
    a.image(str(fig_dir / "05_leakage.png"), width="stretch")
    b.image(str(fig_dir / "11_sequence.png"), width="stretch")

# ---------------------------------------------------------------------------
# Tab 4: about
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("""
**Question.** Can information about a biologic drug, its treatment and the clinical study predict the share of patients
who develop anti-drug antibodies, and which factors are associated with it?

**Data.** Immunogenicity Database Collaborative (immunogenicitydb.org): clinical-trial, therapeutic and sequence tables.
Each patient group gives *n* patients tested and *y* ADA-positive, modelled as **y ~ Binomial(n, p)**.
Pre-treatment baselines removed; repeated timepoints collapsed to one row per group (peak ADA).

**Models.** Binomial GLM (plain and penalised), XGBoost, Bayesian Binomial and **Beta-Binomial** regression (PyMC).
The Beta-Binomial lets each trial group have its own rate around the prediction, which the data clearly need.

**Main findings.**
- For a **new molecule**, no model clearly beats "predict the database average". Most variation is molecule-specific.
- Only the Beta-Binomial gives **honest uncertainty** (≈ 88 % of unseen groups inside its 90 % range; ≈ 50 % for the others).
- Factors the methods agree on: later trial start (more sensitive assays) ↑, protein foreignness (bacterial, chimeric) ↑,
  active-controlled/biosimilar trials ↑, higher dose ↓, co-medication ↓, healthy volunteers ↓.
- Simple sequence features gave no consistent improvement (tested and reported).

**Limitations.** Observational, heterogeneous trials and assays; dose parsed from free text (approximate);
~108 molecules limit what can be learned; associations only.

Code and notebooks: the `immuno-compass` repository (`notebooks/01`–`12`).
""")
