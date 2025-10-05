# ED Crowding — Streamlit Cloud (Train & Predict)

**What it does**
- Loads an existing `ed_crowding_classifier.pkl` *or*
- Trains a compatible model from your CSV under scikit-learn 1.5.x

**Local run**
```
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Streamlit Community Cloud**
1) Push these files to a GitHub repo (root): `streamlit_app.py`, `requirements.txt` (optionally add `ed_crowding_classifier.pkl`).
2) On Streamlit Cloud: New app → select repo/branch → `streamlit_app.py` → Deploy.
3) If you didn't include the model, go to **Train from CSV** tab, upload `Data_Filled.csv`, train, and download a `.pkl`.
