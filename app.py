"""DLM Lab — learn dynamic linear models and the Kalman filter by poking at them.

Run:  streamlit run app.py
"""
import streamlit as st

import ui

st.set_page_config(page_title="DLM Lab", page_icon="📈", layout="wide")

pages = {
    "Start": [st.Page("views/home.py", title="Roadmap", icon="🧭", default=True)],
    "1 · The model": [st.Page("views/model.py", title="Model & data", icon="🧱")],
    "2 · Filtering & smoothing": [
        st.Page("views/filter_step.py", title="Kalman filter, step by step", icon="🔁"),
        st.Page("views/smoother.py", title="Smoothing (RTS)", icon="↩️"),
    ],
    "3 · Estimation": [
        st.Page("views/likelihood.py", title="Likelihood & MLE", icon="⛰️"),
        st.Page("views/diagnostics.py", title="Diagnostics", icon="🩺"),
    ],
    "4 · Using the model": [
        st.Page("views/forecast.py", title="Forecasting", icon="🔮"),
        st.Page("views/discount.py", title="Discount factors", icon="🎚️"),
    ],
    "5 · Bayesian inference": [st.Page("views/bayes.py", title="FFBS + Gibbs sampler", icon="🎲")],
}
pg = st.navigation(pages)

# The sidebar is built once here so every page shares the same data and settings.
st.session_state["ctx"] = ui.sidebar()
pg.run()
