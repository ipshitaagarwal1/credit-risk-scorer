import pickle
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from db import save_assessment, init_db

init_db()


@st.cache_resource
def load_model():
    with open("outputs/credit_risk_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["encoders"], bundle["features"]


model, encoders, feature_order = load_model()
explainer = shap.TreeExplainer(model)

st.sidebar.markdown(f"### 👋 Hi, {st.session_state.username}")
st.sidebar.divider()
st.sidebar.markdown("## 💳 Applicant Profile")

with st.sidebar.expander("👤 Personal Details", expanded=True):
    age = st.slider("Age", 21, 65, 35)
    employment_years = st.slider("Years Employed", 0.0, 40.0, 5.0)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])

with st.sidebar.expander("💰 Financial Profile", expanded=True):
    annual_income = st.number_input("Annual Income ($)", 15000, 300000, 55000, step=1000)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    debt_to_income = st.slider("Debt-to-Income (%)", 0.0, 60.0, 20.0)
    num_open_accounts = st.slider("Open Accounts", 0, 25, 5)
    num_late_payments_2y = st.slider("Late Payments (last 2y)", 0, 15, 0)

with st.sidebar.expander("📄 Loan Details", expanded=True):
    loan_amount = st.number_input("Loan Amount ($)", 1000, 100000, 15000, step=500)
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["debt_consolidation", "credit_card", "home_improvement", "medical", "small_business", "other"]
    )
    loan_term_months = st.selectbox("Loan Term (months)", [36, 60])

loan_to_income_pct = (loan_amount / (annual_income + 1)) * 100
assess_clicked = st.sidebar.button("🔍 Assess Risk", type="primary", use_container_width=True)

st.markdown("# 💳 Risk Assessment")
st.caption("Fill in applicant details in the sidebar, then click Assess Risk.")
st.divider()

if assess_clicked:
    inputs = {
        "age": age, "annual_income": annual_income, "credit_score": credit_score,
        "loan_amount": loan_amount, "employment_years": employment_years,
        "debt_to_income": debt_to_income, "num_open_accounts": num_open_accounts,
        "num_late_payments_2y": num_late_payments_2y, "home_ownership": home_ownership,
        "loan_purpose": loan_purpose, "loan_term_months": loan_term_months,
    }
    row = pd.DataFrame([{
        **inputs,
        "home_ownership": encoders["home_ownership"].transform([home_ownership])[0],
        "loan_purpose": encoders["loan_purpose"].transform([loan_purpose])[0],
        "loan_to_income_pct": loan_to_income_pct,
    }])[feature_order]

    prob = model.predict_proba(row)[0, 1]
    is_high_risk = prob >= 0.5
    decision_label = "HIGH RISK" if is_high_risk else "LOW RISK"

    save_assessment(st.session_state.username, inputs, float(prob), decision_label)

    col1, col2 = st.columns(2)
    with col1:
        gauge_color = "#FF4C4C" if is_high_risk else "#00D980"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"color": "#E8E8E8", "size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#E8E8E8"},
                "bar": {"color": gauge_color},
                "bgcolor": "#1C1F26",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(0,217,128,0.15)"},
                    {"range": [50, 100], "color": "rgba(255,76,76,0.15)"},
                ],
            },
            title={"text": "Default Probability", "font": {"color": "#E8E8E8", "size": 16}},
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E8E8E8"}, height=260,
                           margin=dict(t=50, b=10, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        card_class = "decision-flag" if is_high_risk else "decision-approve"
        label = "🔴 HIGH RISK" if is_high_risk else "🟢 LOW RISK"
        action = "Flag for manual review" if is_high_risk else "Approve"
        st.markdown(f"""
        <div class="{card_class}" style="height: 260px; display: flex; flex-direction: column; justify-content: center;">
            <h2 style="margin:0;">{label}</h2>
            <p style="font-size: 20px; margin-top: 12px;">{action}</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### Why this decision was made")

    sv = explainer(row)
    if len(sv.shape) == 3:
        sv = sv[:, :, 1]

    plt.style.use("dark_background")
    fig2 = plt.figure(facecolor="#0F1117")
    shap.plots.waterfall(sv[0], show=False, max_display=10)
    fig2.patch.set_facecolor("#0F1117")
    for ax in fig2.axes:
        ax.set_facecolor("#0F1117")
    st.pyplot(fig2)
    plt.style.use("default")

    contributions = pd.DataFrame({
        "feature": row.columns, "shap_value": sv.values[0]
    }).sort_values("shap_value", key=abs, ascending=False)

    st.markdown("### Top factors")
    chips_html = ""
    for _, r in contributions.head(5).iterrows():
        if r["shap_value"] > 0:
            chips_html += f'<span class="chip chip-risk">🔴 {r["feature"]} — increases risk ({r["shap_value"]:+.3f})</span>'
        else:
            chips_html += f'<span class="chip chip-safe">🟢 {r["feature"]} — decreases risk ({r["shap_value"]:+.3f})</span>'
    st.markdown(chips_html, unsafe_allow_html=True)

    st.success("✅ This assessment was saved to your history.")
else:
    st.info("👈 Fill in the applicant's details in the sidebar, then click **Assess Risk**.")
