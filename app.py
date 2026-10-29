import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")


@st.cache_resource
def load_artifacts():
    return joblib.load("heart_model.joblib"), joblib.load("app_meta.joblib")


model, meta = load_artifacts()
num_cols, cat_cols = meta["num_cols"], meta["cat_cols"]

st.title("❤️ Heart Disease Predictor")
st.caption("Educational project. Not a medical device and not a substitute for professional diagnosis.")

tab_predict, tab_results = st.tabs(["Predict", "Model results"])

with tab_predict:
    st.subheader("Patient details")
    inputs = {}

    cols = st.columns(3)
    for i, c in enumerate(num_cols):
        lo, hi = meta["num_ranges"][c]
        default = meta["num_defaults"][c]
        with cols[i % 3]:
            if meta["num_is_int"][c]:
                inputs[c] = st.number_input(c, min_value=int(lo), max_value=int(hi), value=int(round(default)), step=1)
            else:
                inputs[c] = st.number_input(c, min_value=float(lo), max_value=float(hi), value=float(default), step=0.1)

    st.markdown("---")
    cols = st.columns(3)
    for i, c in enumerate(cat_cols):
        with cols[i % 3]:
            inputs[c] = st.selectbox(c, meta["choices"][c])

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([inputs])[num_cols + cat_cols]
        proba = float(model.predict_proba(row)[0, 1])
        pred = int(model.predict(row)[0])

        st.subheader("Result")
        m1, m2 = st.columns(2)
        m1.metric("Probability of heart disease", f"{proba:.1%}")
        m2.metric("Prediction", "Heart disease likely" if pred == 1 else "Heart disease unlikely")
        st.progress(min(max(proba, 0.0), 1.0))
        if pred == 1:
            st.error("The model predicts a high likelihood of heart disease. A doctor should be consulted.")
        else:
            st.success("The model predicts a low likelihood of heart disease.")
        with st.expander("Input sent to the model"):
            st.dataframe(row)

with tab_results:
    st.subheader(f"Final model: {meta.get('model_name', 'see notebook')}")
    for title, f in [("Baseline models (test set)", "results/baseline_comparison.csv"),
                     ("Before vs after tuning / ensembling", "results/before_after_improvement.csv")]:
        if os.path.exists(f):
            st.markdown(f"**{title}**")
            st.dataframe(pd.read_csv(f), width="stretch")
    imgs = ["confusion_matrices.png", "roc_curves.png", "baseline_bar.png", "improvement_bar.png",
            "class_distribution.png", "correlation_heatmap.png"]
    cols = st.columns(2)
    shown = 0
    for img in imgs:
        p = os.path.join("results", img)
        if os.path.exists(p):
            with cols[shown % 2]:
                st.image(p, caption=img.replace("_", " ").replace(".png", "").title())
            shown += 1
