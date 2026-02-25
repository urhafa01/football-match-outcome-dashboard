import os
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

st.write("Looking for model at:", MODEL_PATH)

model = joblib.load(MODEL_PATH)

st.title("⚽ Football Match Outcome Predictor")

model = joblib.load("model.joblib")

home_pts = st.slider("Home last5 points", 0.0, 3.0, 1.5)
away_pts = st.slider("Away last5 points", 0.0, 3.0, 1.2)

home_gf = st.slider("Home goals scored (last5)", 0.0, 3.0, 1.5)
away_gf = st.slider("Away goals scored (last5)", 0.0, 3.0, 1.1)

home_ga = st.slider("Home goals conceded (last5)", 0.0, 3.0, 1.0)
away_ga = st.slider("Away goals conceded (last5)", 0.0, 3.0, 1.3)

home_wr = st.slider("Home win rate", 0.0, 1.0, 0.6)
away_wr = st.slider("Away win rate", 0.0, 1.0, 0.4)

data = [[
    home_pts, home_gf, home_ga,
    away_pts, away_gf, away_ga,
    home_pts-away_pts,
    home_gf-away_gf,
    home_ga-away_ga,
    home_wr, away_wr,
    home_wr-away_wr
]]

if st.button("Predict"):
    proba = model.predict_proba(data)[0]
    classes = model.classes_

    st.subheader("Prediction Probabilities")
    for c, p in zip(classes, proba):
        st.write(f"{c}: {p:.2%}")
