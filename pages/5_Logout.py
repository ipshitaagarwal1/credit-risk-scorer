import streamlit as st

st.session_state.authenticated = False
st.session_state.username = None
st.rerun()
