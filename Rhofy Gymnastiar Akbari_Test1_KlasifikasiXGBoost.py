import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

st.set_page_config(page_title="Klasifikasi Cuaca Kolkata", page_icon="🌦️", layout="wide")
st.title("🌦️ Dashboard Klasifikasi Cuaca Kolkata")
st.write("Klasifikasi `weather_code` menggunakan XGBoost Classifier berdasarkan dataset cuaca Kolkata.")

DATA_PATH = Path("kolkata_weather_jan_june_2026.csv")
MODEL_PATH = Path("xgboost_weather_model.pkl")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def train_artifact(path, file_stamp):
    df = load_data(path)
    target = "weather_code"
    drop_cols = [target, "time", "sunrise", "sunset", "city", "latitude", "longitude"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df[target].astype(int)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

    labels = sorted(y.unique())
    to_idx = {v:i for i,v in enumerate(labels)}
    from_idx = {i:v for v,i in to_idx.items()}
    y_train_enc = y_train.map(to_idx)
    y_test_enc = y_test.map(to_idx)
    n = len(labels)
    model = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        objective="multi:softprob" if n > 2 else "binary:logistic",
        num_class=n if n > 2 else None,
        eval_metric="mlogloss" if n > 2 else "logloss",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train_enc)
    pred = model.predict(X_test).astype(int)
    proba = model.predict_proba(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test_enc, pred),
        "Precision (weighted)": precision_score(y_test_enc, pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_test_enc, pred, average="weighted", zero_division=0),
        "F1-score (weighted)": f1_score(y_test_enc, pred, average="weighted", zero_division=0)
    }
    try:
        y_bin = label_binarize(y_test_enc, classes=list(range(n)))
        metrics["ROC-AUC (weighted OvR)"] = roc_auc_score(y_bin, proba, multi_class="ovr", average="weighted")
    except Exception:
        metrics["ROC-AUC (weighted OvR)"] = np.nan
    cm = confusion_matrix(y_test_enc, pred, labels=list(range(n)))
    report = pd.DataFrame({
        "weather_code": labels,
        "jumlah": [int((y == c).sum()) for c in labels],
        "proporsi (%)": [round(float((y == c).mean()*100), 2) for c in labels]
    })
    return df, X, y, X_test, y_test_enc, pred, proba, labels, model, imputer, metrics, cm, report, to_idx, from_idx

if not DATA_PATH.exists():
    st.error(f"File dataset tidak ditemukan: {DATA_PATH.resolve()}. Letakkan CSV di folder yang sama dengan app.py.")
    st.stop()

df = load_data(str(DATA_PATH))
drop_cols = ["weather_code", "time", "sunrise", "sunset", "city", "latitude", "longitude"]
X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns]).apply(pd.to_numeric, errors="coerce")
X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
try:
    (df, X, y, X_test, y_test_enc, pred, proba, labels, model, imputer,
     metrics, cm, class_report, to_idx, from_idx) = train_artifact(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as e:
    st.exception(e)
    st.stop()

st.sidebar.header("Navigasi")
section = st.sidebar.radio("Pilih bagian", ["Ringkasan & EDA", "Evaluasi Model", "Coba Prediksi"])

if section == "Ringkasan & EDA":
    c1,c2,c3 = st.columns(3)
    c1.metric("Jumlah observasi", f"{len(df):,}")
    c2.metric("Jumlah fitur model", str(X.shape[1]))
    c3.metric("Jumlah kelas", str(len(labels)))
    st.subheader("Struktur dataset")
    st.write(f"Rentang tanggal: **{df['time'].min()}** hingga **{df['time'].max()}**")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Proporsi kelas target")
    st.dataframe(class_report, use_container_width=True)
    fig, ax = plt.subplots(figsize=(9,4))
    sns.countplot(data=df, x="weather_code", order=labels, ax=ax)
    ax.set_title("Distribusi Weather Code")
    ax.set_xlabel("Weather code")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)
    st.subheader("Distribusi fitur numerik")
    feature = st.selectbox("Pilih fitur", X.columns.tolist())
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df[feature].dropna(), bins=25, kde=True, ax=ax)
    ax.set_title(f"Distribusi {feature}")
    st.pyplot(fig)
    st.subheader("Heatmap korelasi")
    fig, ax = plt.subplots(figsize=(12,7))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Korelasi antar variabel numerik")
    st.pyplot(fig)
    st.caption("Catatan: weather_code dan fitur cuaca adalah pengamatan pada hari yang sama. Dashboard ini mendemonstrasikan klasifikasi kondisi teramati, bukan peramalan masa depan.")

elif section == "Evaluasi Model":
    st.subheader("Metrik Evaluasi XGBoost")
    st.dataframe(pd.DataFrame({"Metrik": list(metrics.keys()), "Nilai": list(metrics.values())}).style.format({"Nilai":"{:.4f}"}), use_container_width=True)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(9,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig)
    st.subheader("ROC Curve One-vs-Rest")
    y_bin = label_binarize(y_test_enc, classes=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(8,6))
    plotted = 0
    for i, label in enumerate(labels):
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        ax.plot(fpr, tpr, label=f"Kelas {label} (AUC={auc(fpr,tpr):.2f})")
        plotted += 1
    ax.plot([0,1],[0,1],"k--",label="Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("ROC Curve Multiclass OvR")
    st.pyplot(fig)
    st.subheader("Feature Importance")
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8,5))
    imp.plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    st.pyplot(fig)

else:
    st.subheader("Form Prediksi Manual")
    st.write("Masukkan nilai fitur cuaca. Model akan memilih weather_code dengan probabilitas tertinggi.")
    values = {}
    defaults = X.median(numeric_only=True)
    with st.form("prediction_form"):
        cols = st.columns(2)
        for i, col in enumerate(X.columns):
            with cols[i % 2]:
                vals = pd.to_numeric(X[col], errors="coerce").dropna()
                lo = float(vals.min()) if not vals.empty else 0.0
                hi = float(vals.max()) if not vals.empty else 100.0
                default = float(defaults[col]) if pd.notna(defaults[col]) else 0.0
                values[col] = st.number_input(col, min_value=lo, max_value=hi, value=min(max(default,lo),hi), format="%.3f")
        submitted = st.form_submit_button("Prediksi weather_code")
    if submitted:
        sample = pd.DataFrame([values], columns=X.columns)
        sample_imp = pd.DataFrame(imputer.transform(sample), columns=X.columns)
        p = model.predict_proba(sample_imp)[0]
        pred_idx = int(np.argmax(p))
        predicted_code = from_idx[pred_idx]
        st.success(f"Prediksi weather_code: {predicted_code}")
        st.metric("Probabilitas model untuk kelas terpilih", f"{p[pred_idx]*100:.2f}%")
        st.dataframe(pd.DataFrame({"weather_code": labels, "probabilitas (%)": np.round(p*100,2)}).sort_values("probabilitas (%)", ascending=False), use_container_width=True)

st.sidebar.caption("Model: XGBoost Classifier • test split 80:20 • random_state=42")