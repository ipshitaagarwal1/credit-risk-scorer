import streamlit as st

st.markdown('<span class="hero-badge">EXPLAINABLE AI · CREDIT RISK</span>', unsafe_allow_html=True)
st.markdown("# 💳 Explainable Credit Risk Scorer")
st.markdown(
    "### Predict loan default risk — and see exactly why, in plain language."
)
st.write("")
st.write("")

_, mid, _ = st.columns([1, 1, 1])
with mid:
    if st.button("Get Started →", type="primary", use_container_width=True):
        st.switch_page("pages/1_Login.py")

st.write("")
st.caption("Built with XGBoost, SHAP, LIME, and Streamlit. Not affiliated with any real financial institution — this is a portfolio project using synthetic data.")

