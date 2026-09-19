"""Streamlit version of the heart disease predictor.

Run:  pip install streamlit scikit-learn pandas numpy
      streamlit run app.py
Expects data.csv next to this file.
"""
import pandas as pd
import streamlit as st

import heart_disease_pipeline as hp

st.set_page_config(page_title="Heart disease risk predictor", layout="wide")


@st.cache_resource(show_spinner="Training models on the Cleveland dataset...")
def load():
    return hp.train("data.csv")


art = load()
typ = hp.typical_patient(art["raw"])
CHOICES = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "No symptoms"},
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {1: "Normal", 2: "Fixed defect", 3: "Reversible defect"},
}
LABEL = {"age": "Age (years)", "sex": "Sex", "cp": "Chest pain type",
         "trestbps": "Resting blood pressure (mm Hg)", "chol": "Serum cholesterol (mg/dl)",
         "fbs": "Fasting blood sugar > 120 mg/dl", "restecg": "Resting ECG",
         "thalach": "Maximum heart rate (bpm)", "exang": "Exercise-induced angina",
         "oldpeak": "ST depression", "slope": "Peak exercise ST slope",
         "ca": "Major vessels colored", "thal": "Thallium stress test"}
NUM = {"age": (20, 90, 1), "trestbps": (80, 220, 1), "chol": (100, 600, 1),
       "thalach": (60, 220, 1), "oldpeak": (0.0, 7.0, 0.1)}

st.title("Heart disease risk predictor")
st.caption("Educational project. Not a medical device and not a substitute for clinical judgment.")

left, right = st.columns([3, 2])
rec = {}
with left:
    cols = st.columns(2)
    for i, f in enumerate(hp.FEATURES):
        with cols[i % 2]:
            if f in NUM:
                lo, hi, step = NUM[f]
                default = float(typ[f]) if f == "oldpeak" else int(typ[f])
                rec[f] = st.slider(LABEL[f], lo, hi, default, step)
            else:
                opts = list(CHOICES[f])
                rec[f] = st.radio(LABEL[f], opts, index=opts.index(typ[f]),
                                  format_func=CHOICES[f].get, horizontal=True)

with right:
    choice = st.radio("Model", ["Random Forest", "Logistic Regression", "MLP"], horizontal=True)
    p = hp.predict_record(art, rec, choice)
    st.metric("Estimated probability of heart disease", f"{p:.0%}")
    st.progress(p)
    st.subheader("Heart disease indicated" if p >= .5 else "Heart disease not indicated")
    if .35 <= p <= .65:
        st.warning("Borderline: the model is close to undecided.")
    st.write("**All models**")
    st.dataframe(pd.DataFrame(
        [{"Model": m, "Probability": f"{hp.predict_record(art, rec, m):.0%}",
          "Verdict": "Indicated" if hp.predict_record(art, rec, m) >= .5 else "Not indicated"}
         for m in ["Random Forest", "Logistic Regression", "MLP"]]), hide_index=True)
    # occlusion sensitivity: swap each value for the typical patient's value
    drv = sorted(((f, p - hp.predict_record(art, {**rec, f: typ[f]}, choice)) for f in hp.FEATURES),
                 key=lambda t: -abs(t[1]))[:6]
    st.write("**What is driving this prediction** (change in probability vs a typical patient)")
    st.bar_chart(pd.Series({LABEL[f]: d * 100 for f, d in drv if abs(d) >= .005}))
