import streamlit as st
from db import create_user, verify_user

st.markdown("# 🔐 Login or Create an Account")
st.write("")

tab_login, tab_register = st.tabs(["Login", "Create Account"])

with tab_login:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)
        if submitted:
            if verify_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.switch_page("pages/2_Risk_Assessment.py")
            else:
                st.error("Incorrect username or password.")

with tab_register:
    with st.form("register_form"):
        new_username = st.text_input("Choose a username")
        new_password = st.text_input("Choose a password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)
        if submitted:
            if not new_username or not new_password:
                st.error("Please fill in all fields.")
            elif len(new_password) < 4:
                st.error("Password should be at least 4 characters.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, msg = create_user(new_username, new_password)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
