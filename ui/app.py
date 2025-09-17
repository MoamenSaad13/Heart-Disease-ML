# ui/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ─────────────────────────────
# Page setup
# ─────────────────────────────
st.set_page_config(page_title="Heart Disease Risk - ML Demo", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter your clinical parameters to get an instant prediction from the trained ML pipeline.")

# ─────────────────────────────
# Model loading (final → baseline fallback)
# ─────────────────────────────
PROJECT_ROOT_CANDIDATES = [
    Path(__file__).resolve().parents[1],
    Path.cwd(),
    Path.cwd().parent
]

def resolve_models_dir():
    for cand in PROJECT_ROOT_CANDIDATES:
        models_dir = cand / "models"
        if models_dir.exists():
            return models_dir
    # fallback: create under first candidate
    models_dir = PROJECT_ROOT_CANDIDATES[0] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir

MODELS_DIR = resolve_models_dir()
FINAL_MODEL = MODELS_DIR / "final_model.pkl"
BASELINE_MODEL = MODELS_DIR / "best_baseline.pkl"

@st.cache_resource
def load_pipeline():
    # prefer final model if exists
    if FINAL_MODEL.exists():
        st.success(f"Loaded model: {FINAL_MODEL.name}")
        return joblib.load(FINAL_MODEL)
    if BASELINE_MODEL.exists():
        st.info(f"Loaded model: {BASELINE_MODEL.name}")
        return joblib.load(BASELINE_MODEL)
    return None

pipeline = load_pipeline()

# ─────────────────────────────
# Helper: align inputs to pipeline expected columns
# ─────────────────────────────
def get_expected_feature_names(pipe):
    """
    Try to infer the exact input feature names the pipeline expects.
    Priority:
      1) pipe.named_steps['prep'].feature_names_in_
      2) pipe.feature_names_in_
    """
    expected = None
    try:
        expected = list(pipe.named_steps["prep"].feature_names_in_)
    except Exception:
        pass
    if expected is None:
        try:
            expected = list(pipe.feature_names_in_)
        except Exception:
            pass
    return expected

def align_to_pipeline_columns(df: pd.DataFrame, pipe):
    """
    Ensure df has EXACTLY the same columns (names & order) the pipeline was trained on.
    Missing columns are added with NaN (imputers will handle them). Extra columns are dropped.
    """
    expected = get_expected_feature_names(pipe)
    if expected is None:
        raise RuntimeError(
            "Cannot infer expected input columns from the pipeline. "
            "Ensure the saved model is a Pipeline with a preprocessing step named 'prep', "
            "or that it was fitted with feature_names_in_."
        )

    # map for case-insensitive matching
    df_map_lower = {c.lower(): c for c in df.columns}
    expected_lower = [c.lower() for c in expected]

    aligned_row = {}
    for exp_name, exp_lower in zip(expected, expected_lower):
        if exp_lower in df_map_lower:
            aligned_row[exp_name] = df[df_map_lower[exp_lower]].iloc[0]
        else:
            aligned_row[exp_name] = np.nan  # imputer will fill

    aligned_df = pd.DataFrame([aligned_row], columns=expected)

    # helpful debug
    provided_set = set([c.lower() for c in df.columns])
    missing = [e for e in expected if e.lower() not in provided_set]
    extra = [c for c in df.columns if c.lower() not in [e.lower() for e in expected]]

    if missing:
        st.info(f"Adding missing columns (filled with NaN for imputation): {missing}")
    if extra:
        st.info(f"Dropping extra columns not used by the model: {extra}")

    # try to cast numerics
    for col in aligned_df.columns:
        aligned_df[col] = pd.to_numeric(aligned_df[col], errors="ignore")
    return aligned_df

# ─────────────────────────────
# Sidebar: model/debug info
# ─────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Info")
    if pipeline is None:
        st.warning("No model found. Train it first (see notebooks) to create `models/final_model.pkl` or `best_baseline.pkl`.")
    else:
        exp = get_expected_feature_names(pipeline)
        if exp:
            st.caption("Expected feature columns:")
            st.code("\n".join(exp), language="text")
        else:
            st.caption("Could not introspect expected columns. Make sure your pipeline has a 'prep' step or feature_names_in_.")

# ─────────────────────────────
# Inputs
# (Use the common Heart Disease attributes — names match Kaggle-style schema)
# ─────────────────────────────
with st.expander("Input Fields", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=120, value=54)
        sex = st.selectbox("Sex (1=male, 0=female)", options=[1, 0], index=0)
        cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3], index=0)
        trestbps = st.number_input("Resting Blood Pressure", min_value=70, max_value=260, value=130)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=80, max_value=700, value=246)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1/0)", options=[1, 0], index=1)
    with col2:
        restecg = st.selectbox("Resting ECG Results (0-2)", options=[0, 1, 2], index=1)
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina (1/0)", options=[1, 0], index=1)
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", options=[0, 1, 2], index=2)
        ca = st.selectbox("Number of Major Vessels (0-3) Colored by Fluoroscopy", options=[0, 1, 2, 3], index=0)
        thal = st.selectbox("Thalassemia (1=normal,2=fixed defect,3=reversible defect)", options=[1, 2, 3], index=2)

input_df = pd.DataFrame([{
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
}])

st.subheader("Your Inputs")
st.dataframe(input_df, use_container_width=True)

# ─────────────────────────────
# Predict
# ─────────────────────────────
if pipeline is None:
    st.warning("Model file not found. Train the model via notebooks (or run `python run_all.py`) to create `models/final_model.pkl`.")
else:
    if st.button("Predict Risk", type="primary"):
        try:
            X_input = align_to_pipeline_columns(input_df, pipeline)
            proba = float(pipeline.predict_proba(X_input)[0, 1])
            pred = int(proba >= 0.5)

            st.subheader("Prediction")
            st.metric("Risk Probability", f"{proba:.2%}")
            st.write("**Predicted Class**:", "Heart Disease" if pred == 1 else "No Heart Disease")

            with st.expander("Debug: Aligned Input Row"):
                st.write(X_input)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
