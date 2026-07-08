"""
Streamlit demo for the Explainable Credit Risk Scorer.

A loan officer types in applicant details, gets an instant risk score,
plus a SHAP waterfall chart explaining exactly why.

Run locally:
    streamlit run app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Explainable Credit Risk Scorer", layout="wide")


@st.cache_resource
def load_model():
    with open("outputs/credit_risk_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["encoders"], bundle["features"]


model, encoders, feature_order = load_model()
explainer = shap.TreeExplainer(model)

st.title("💳 Explainable Credit Risk Scorer")
st.caption("Enter applicant details to see the predicted default risk and exactly why the model made that call.")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 21, 65, 35)
    annual_income = st.number_input("Annual Income ($)", 15000, 300000, 55000, step=1000)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    loan_amount = st.number_input("Loan Amount ($)", 1000, 100000, 15000, step=500)

with col2:
    employment_years = st.slider("Years Employed", 0.0, 40.0, 5.0)
    debt_to_income = st.slider("Debt-to-Income (%)", 0.0, 60.0, 20.0)
    num_open_accounts = st.slider("Open Accounts", 0, 25, 5)
    num_late_payments_2y = st.slider("Late Payments (last 2y)", 0, 15, 0)

with col3:
    home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["debt_consolidation", "credit_card", "home_improvement", "medical", "small_business", "other"]
    )
    loan_term_months = st.selectbox("Loan Term (months)", [36, 60])

loan_to_income_pct = (loan_amount / (annual_income + 1)) * 100

if st.button("Assess Risk", type="primary"):
    row = pd.DataFrame([{
        "age": age,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "debt_to_income": debt_to_income,
        "num_open_accounts": num_open_accounts,
        "num_late_payments_2y": num_late_payments_2y,
        "home_ownership": encoders["home_ownership"].transform([home_ownership])[0],
        "loan_purpose": encoders["loan_purpose"].transform([loan_purpose])[0],
        "loan_term_months": loan_term_months,
        "loan_to_income_pct": loan_to_income_pct,
    }])[feature_order]

    prob = model.predict_proba(row)[0, 1]
    decision = "🔴 HIGH RISK — Flag for review" if prob >= 0.5 else "🟢 LOW RISK — Approve"

    st.subheader(f"Predicted Default Probability: {prob:.1%}")
    st.markdown(f"### Decision: {decision}")

    sv = explainer(row)
    if len(sv.shape) == 3:
        sv = sv[:, :, 1]

    st.markdown("#### Why this decision was made")
    fig = plt.figure()
    shap.plots.waterfall(sv[0], show=False, max_display=10)
    st.pyplot(fig)

    contributions = pd.DataFrame({
        "feature": row.columns,
        "shap_value": sv.values[0]
    }).sort_values("shap_value", key=abs, ascending=False)

    st.markdown("#### Top factors (plain language)")
    for _, r in contributions.head(3).iterrows():
        direction = "increases" if r["shap_value"] > 0 else "decreases"
        st.write(f"- **{r['feature']}** {direction} this applicant's risk (impact: {r['shap_value']:+.3f})")
