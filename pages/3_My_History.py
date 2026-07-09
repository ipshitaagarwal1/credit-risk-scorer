import streamlit as st
import plotly.graph_objects as go
from db import get_user_history, init_db

init_db()

st.markdown("# 📊 My Assessment History")
st.caption(f"Logged in as **{st.session_state.username}**")
st.divider()

df = get_user_history(st.session_state.username)

if df.empty:
    st.info("You haven't run any assessments yet. Go to **Risk Assessment** to run your first one.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Assessments", len(df))
    col2.metric("Average Risk", f"{df['risk_probability'].mean():.1%}")
    col3.metric("High Risk Flagged", f"{(df['decision'] == 'HIGH RISK').sum()} / {len(df)}")

    st.write("")
    st.markdown("### Risk over time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["risk_probability"] * 100,
        mode="lines+markers", line={"color": "#00D9C0"}, marker={"size": 8}
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1C1F26",
        font={"color": "#E8E8E8"}, yaxis_title="Default Probability (%)",
        height=320, margin=dict(t=20, b=10, l=30, r=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Full history")
    display_df = df[[
        "timestamp", "age", "annual_income", "credit_score", "loan_amount",
        "debt_to_income", "risk_probability", "decision"
    ]].copy()
    display_df["risk_probability"] = (display_df["risk_probability"] * 100).round(1).astype(str) + "%"
    st.dataframe(display_df, use_container_width=True, hide_index=True)
