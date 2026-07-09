import streamlit as st

st.markdown('<span class="hero-badge">EXPLAINABLE AI · CREDIT RISK</span>', unsafe_allow_html=True)
st.markdown("# 💳 Explainable Credit Risk Scorer")
st.markdown(
    "### Predict loan default risk — and see exactly why, in plain language."
)
st.write("")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🎯 0.92 ROC-AUC</h3>
        <p style="color:#9AA0AC;">XGBoost model benchmarked against Random Forest on 6,000 applicant profiles.</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>🔍 SHAP + LIME</h3>
        <p style="color:#9AA0AC;">Every prediction comes with a human-readable explanation, not just a score.</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>📊 Full History</h3>
        <p style="color:#9AA0AC;">Every assessment you run is saved to your account so you can track it over time.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

_, mid, _ = st.columns([1, 1, 1])
with mid:
    if st.button("Get Started →", type="primary", use_container_width=True):
        st.switch_page("pages/1_Login.py")

st.write("")
st.caption("Built with XGBoost, SHAP, LIME, and Streamlit. Not affiliated with any real financial institution — this is a portfolio project using synthetic data.")
