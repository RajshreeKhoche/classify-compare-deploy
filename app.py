from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Bean Variety Classifier Lab", page_icon="🫘", layout="wide")

MODEL_DIR = Path(__file__).parent / "model"
TARGET_COL = "Class"
FEATURE_COLS = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
    "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
    "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2",
    "ShapeFactor3", "ShapeFactor4",
]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


@st.cache_resource
def load_model(file_name):
    return joblib.load(MODEL_DIR / file_name)


@st.cache_resource
def load_label_encoder():
    return joblib.load(MODEL_DIR / "label_encoder.joblib")


def score_model(model, X, y_true):
    """Return predictions and the six metrics (no retraining, saved model only)."""
    pred = model.predict(X)
    proba = model.predict_proba(X)
    try:
        auc = roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro",
            labels=list(range(proba.shape[1])),
        )
    except ValueError:
        auc = np.nan  # happens when a class is missing from the uploaded file
    metrics = {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": auc,
        "Precision": precision_score(y_true, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
    }
    return pred, metrics


st.title("🫘 Bean Variety Classifier Lab")
st.write(
    "Upload the **test CSV** (16 feature columns plus the `Class` column), pick one of the "
    "five saved models, and see how it performs. Models are loaded from disk; "
    "nothing is retrained on your upload."
)

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Step 1: upload test CSV", type=["csv"])
    chosen_name = st.selectbox("Step 2: choose a model", list(MODEL_FILES.keys()))
    st.caption("Metrics: AUC is one-vs-rest (macro); precision, recall and F1 are weighted.")

if uploaded is None:
    st.info("Upload a CSV in the sidebar to begin. Required columns:")
    st.code(", ".join(FEATURE_COLS + [TARGET_COL]))
    st.stop()

data = pd.read_csv(uploaded, encoding="utf-8-sig")

missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in data.columns]
if missing:
    st.error(f"The uploaded file is missing these columns: {missing}")
    st.stop()

label_encoder = load_label_encoder()
unknown = set(data[TARGET_COL].unique()) - set(label_encoder.classes_)
if unknown:
    st.error(f"Unknown class labels in `{TARGET_COL}`: {sorted(unknown)}")
    st.stop()

X_upload = data[FEATURE_COLS]
y_upload = label_encoder.transform(data[TARGET_COL])
class_names = list(label_encoder.classes_)

tab_single, tab_all, tab_data = st.tabs(["Selected model", "All models compared", "Uploaded data"])

with tab_single:
    st.subheader(f"Results for {chosen_name}")
    model = load_model(MODEL_FILES[chosen_name])
    pred, metrics = score_model(model, X_upload, y_upload)

    cols = st.columns(6)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, "n/a" if np.isnan(value) else f"{value:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_upload, pred, labels=list(range(len(class_names))))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            y_upload, pred, labels=list(range(len(class_names))),
            target_names=class_names, zero_division=0, output_dict=True,
        )
        st.dataframe(pd.DataFrame(report).T.round(3))

with tab_all:
    st.subheader("All five models on the uploaded data")
    rows = {}
    for name, file_name in MODEL_FILES.items():
        _, m = score_model(load_model(file_name), X_upload, y_upload)
        rows[name] = m
    comparison = pd.DataFrame(rows).T.round(4)
    st.dataframe(comparison)
    st.caption(f"Highest MCC on this upload: {comparison['MCC'].idxmax()}")

with tab_data:
    st.write(f"{data.shape[0]} rows, {data.shape[1]} columns")
    st.dataframe(data.head(50))