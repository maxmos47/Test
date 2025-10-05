import os
import pickle
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="ED Crowding (1–5): Train & Predict", page_icon="🏥", layout="wide")
st.title("🏥 ED Crowding Classifier (1–5) — Train & Predict")
st.caption("Load an existing model or train a new one from CSV (compatible with scikit-learn 1.5.x).")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    dr = ['DR_Score1','DR_Score2','DR_Score3']
    ns = ['NS_Score1','NS_Score2','NS_Score3']
    for c in dr+ns:
        if c not in df.columns: df[c] = np.nan
    df['Crowding_Score'] = df[dr+ns].mean(axis=1).round().clip(1,5)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(['Date','Shift'], na_position='last').reset_index(drop=True)

    base = [c for c in [
        'Patients','ED_Occupancy','WaitingTime','Admit','Discharge',
        'Doctor','Nurse','PN','EDWIN','NEDOCS','AT_Mean','AT_Max',
        'Holiday','WaitingAdmit','LWBS'
    ] if c in df.columns]

    for l in [1,2,3]:
        for c in base+['Crowding_Score']:
            df[f"{c}_Lag{l}"] = df[c].shift(l)

    for c in [x for x in ['Patients','EDWIN','NEDOCS','WaitingTime','Admit','Discharge'] if x in df.columns]:
        df[f"{c}_Roll3Mean"] = df[c].rolling(window=3, min_periods=1).mean().shift(1)

    df['dow'] = df['Date'].dt.dayofweek
    return df

def train_pipeline(df: pd.DataFrame):
    df = build_features(df)
    cat_cols = [c for c in ['Day','Shift'] if c in df.columns]

    base = [c for c in [
        'Patients','ED_Occupancy','WaitingTime','Admit','Discharge',
        'Doctor','Nurse','PN','EDWIN','NEDOCS','AT_Mean','AT_Max',
        'Holiday','WaitingAdmit','LWBS'
    ] if c in df.columns]
    features = base             + [f"{c}_Lag{l}" for c in base+['Crowding_Score'] for l in [1,2,3]]             + [f"{c}_Roll3Mean" for c in ['Patients','EDWIN','NEDOCS','WaitingTime','Admit','Discharge'] if f"{c}_Roll3Mean" in df.columns]             + ['dow'] + cat_cols
    features = [c for c in features if c in df.columns]

    dfm = df.dropna(subset=['Crowding_Score']+features).copy()

    n = len(dfm)
    split_idx = int(n * 0.8)
    train_df = dfm.iloc[:split_idx]
    test_df  = dfm.iloc[split_idx:]
    X_train = train_df[features]; y_train = train_df['Crowding_Score'].astype(int)
    X_test  = test_df[features];  y_test  = test_df['Crowding_Score'].astype(int)

    transformers = []
    if cat_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols))
    prep = ColumnTransformer(transformers=transformers, remainder='passthrough')

    model = RandomForestClassifier(
        n_estimators=500, min_samples_split=4,
        class_weight='balanced', random_state=42
    )
    pipe = Pipeline([('prep', prep), ('model', model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    rpt = classification_report(y_test, y_pred, zero_division=0)
    cm  = confusion_matrix(y_test, y_pred)

    blob = pickle.dumps({'pipeline': pipe, 'artifacts': {'feature_candidates': features, 'cat_cols': cat_cols}})
    return pipe, features, cat_cols, rpt, cm, blob

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return obj['pipeline'], obj['artifacts']['feature_candidates'], obj['artifacts'].get('cat_cols', [])

MODEL_PATH = os.environ.get("MODEL_PATH", "ed_crowding_classifier.pkl")
model = None; FEATURES=[]; CAT_COLS=[]
try:
    model, FEATURES, CAT_COLS = load_model(MODEL_PATH)
    st.success(f"Loaded model ✓ ({MODEL_PATH}) — {len(FEATURES)} features")
except Exception as e:
    st.warning(f"Could not load '{MODEL_PATH}': {e}")

tab_pred, tab_batch, tab_train = st.tabs(["🔮 Predict (single)", "📄 Batch CSV", "🛠️ Train from CSV"])

with tab_train:
    st.subheader("Train a new model from CSV")
    up = st.file_uploader("Upload your Data_Filled.csv", type=["csv"])
    if up is not None and st.button("Train now", type="primary"):
        df = pd.read_csv(up)
        with st.spinner("Training..."):
            pipe, feats, cats, rpt, cm, blob = train_pipeline(df)
        st.success("Training done ✓")
        st.text("Classification report:")
        st.code(rpt)
        st.text("Confusion matrix:")
        st.write(cm)

        st.download_button("⬇️ Download trained model (.pkl)", data=blob,
                           file_name="ed_crowding_classifier.pkl", mime="application/octet-stream")

        # Keep in session for immediate use
        st.session_state['pipe']=pipe; st.session_state['features']=feats; st.session_state['cats']=cats
        st.info("Model loaded into session. Switch to Predict tab to use it now.")

def ensure_model():
    if model is not None and len(FEATURES)>0:
        return model, FEATURES, CAT_COLS
    if 'pipe' in st.session_state:
        return st.session_state['pipe'], st.session_state['features'], st.session_state.get('cats', [])
    raise RuntimeError("No model available. Load PKL or train from CSV.")

with tab_pred:
    st.subheader("Single Prediction")
    try:
        m, feats, cats = ensure_model()
        cols = st.columns(3)
        vals: Dict[str, Any] = {}
        for i, f in enumerate(feats):
            with cols[i % 3]:
                v = st.text_input(f, value="")
                if v.strip()=="":
                    vals[f]=0
                else:
                    try: vals[f]=float(v)
                    except ValueError: vals[f]=v
        if st.button("Predict crowding level", type="primary"):
            row = {f: vals.get(f,0) for f in feats}
            X = pd.DataFrame([row], columns=feats)
            pred = m.predict(X)[0]
            st.metric("Predicted ED crowding level", int(pred))
            with st.expander("Debug payload"): st.json(row)
    except Exception as e:
        st.info("Load a model (PKL) or train from CSV first.")
        st.error(f"Prediction unavailable: {e}")

with tab_batch:
    st.subheader("Batch Prediction (CSV)")
    try:
        m, feats, cats = ensure_model()
        st.write("Upload a CSV containing exactly these columns (any order):")
        st.code(", ".join(feats), language="text")
        upb = st.file_uploader("Upload CSV", type=["csv"], key="batch")
        if upb is not None:
            dfb = pd.read_csv(upb)
            missing = [c for c in feats if c not in dfb.columns]
            if missing:
                st.warning(f"Missing required columns: {missing}")
            else:
                X = dfb.reindex(columns=feats).fillna(0)
                preds = m.predict(X).astype(int)
                out = dfb.copy()
                out["Predicted_Crowding_Level"] = preds
                st.dataframe(out.head(20), use_container_width=True)
                st.download_button("⬇️ Download results CSV",
                    data=out.to_csv(index=False).encode("utf-8-sig"),
                    file_name="ed_crowding_predictions.csv", mime="text/csv")
    except Exception as e:
        st.info("Load a model (PKL) or train from CSV first.")
        st.error(f"Batch prediction unavailable: {e}")
