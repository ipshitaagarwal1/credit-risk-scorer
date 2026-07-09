"""
Explainable Credit Risk Scorer — multi-page app entry point.

Run locally:
    streamlit run app.py
"""

import streamlit as st
from db import init_db

st.set_page_config(page_title="Explainable Credit Risk Scorer", page_icon="💳", layout="wide")
init_db()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None

# Shared dark-theme card/chip CSS used across all pages
st.markdown("""
<style>
    .metric-card { background-color: #1C1F26; border: 1px solid #2A2E38; border-radius: 14px; padding: 24px; text-align: center; }
    .decision-approve { background-color: rgba(0, 217, 128, 0.12); border: 1px solid #00D980; border-radius: 14px; padding: 24px; text-align: center; }
    .decision-flag { background-color: rgba(255, 76, 76, 0.12); border: 1px solid #FF4C4C; border-radius: 14px; padding: 24px; text-align: center; }
    .chip { display: inline-block; padding: 8px 14px; border-radius: 20px; margin: 4px 6px 4px 0; font-size: 14px; }
    .chip-risk { background-color: rgba(255, 76, 76, 0.15); border: 1px solid #FF4C4C; color: #FF8080; }
    .chip-safe { background-color: rgba(0, 217, 128, 0.15); border: 1px solid #00D980; color: #5CFFB8; }
    .hero-badge { display: inline-block; padding: 6px 14px; border-radius: 20px; background: rgba(0,217,192,0.12); border: 1px solid #00D9C0; color: #00D9C0; font-size: 13px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

if not st.session_state.authenticated:
    landing = st.Page("pages/0_Welcome.py", title="Welcome", icon="🏠", default=True)
    login = st.Page("pages/1_Login.py", title="Login / Register", icon="🔐")
    pg = st.navigation([landing, login])
else:
    dashboard = st.Page("pages/2_Risk_Assessment.py", title="Risk Assessment", icon="💳", default=True)
    history = st.Page("pages/3_My_History.py", title="My History", icon="📊")
    how_it_works = st.Page("pages/4_How_It_Works.py", title="How It Works", icon="ℹ️")
    logout = st.Page("pages/5_Logout.py", title="Logout", icon="🚪")
    pg = st.navigation([dashboard, history, how_it_works, logout])

pg.run()
