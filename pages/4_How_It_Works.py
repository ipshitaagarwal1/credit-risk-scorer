import streamlit as st

st.markdown("# ℹ️ How It Works")
st.write("")

st.markdown("""
### The Model
Two models were trained and compared on 6,000 synthetic loan applicant records: **Random Forest** and **XGBoost**.
The one with the higher ROC-AUC score (0.92) was selected as the production model.

### Why Explainability Matters
A model that just says "rejected" isn't good enough for lending decisions — regulators and loan officers need
to know *why*. This app uses two explainability techniques on every single prediction:

- **SHAP (SHapley Additive exPlanations)** — based on game theory, calculates each feature's fair contribution
  to a specific prediction. The waterfall chart you see after each assessment shows exactly which factors pushed
  risk up or down, and by how much.
- **LIME (Local Interpretable Model-agnostic Explanations)** — perturbs the applicant's data thousands of times
  and fits a simple, interpretable model around just that one prediction, producing an easy-to-read local explanation.

### Your Data
Every assessment you run is saved to your account (visible under **My History**) so you can track patterns
over time. This is a portfolio/demo project using a synthetic dataset — no real applicant data is used or stored.
""")
