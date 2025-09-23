# svm_model.py
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# Constantes / rutas
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "cell_images_features.csv")  # crea/coloca tu CSV aquí
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_DIR = os.path.join(BASE_DIR, "static", "images")
MODEL_PATH = os.path.join(MODEL_DIR, "svm_cell_model.joblib")

RANDOM_STATE = 42  # semilla para reproducibilidad

# Nombre de las columnas esperadas (orden consistente)
FEATURE_COLUMNS = [
    "textura",       # numérico
    "contraste",     # numérico
    "forma_nucleo",  # numérico
    "area_cell",     # numérico
    "densidad"       # numérico
]
TARGET_NAME = "target"  # columna objetivo (0 = No cancerosa, 1 = Cancerosa)

def load_dataset():
    """
    Carga dataset desde DATA_PATH. Si no existe, genera un dataset sintético de ejemplo.
    Debe contener las columnas FEATURE_COLUMNS y TARGET_NAME con valores 0/1.
    """
    if not os.path.exists(DATA_PATH):
        # Generar dataset sintético de ejemplo (200 muestras)
        rng = np.random.RandomState(RANDOM_STATE)
        n = 400
        textura = rng.normal(loc=0.5, scale=0.12, size=n)
        contraste = rng.normal(loc=0.5, scale=0.15, size=n)
        forma_nucleo = rng.normal(loc=0.5, scale=0.1, size=n)
        area_cell = rng.normal(loc=100.0, scale=30.0, size=n)
        densidad = rng.normal(loc=0.6, scale=0.2, size=n)
        # crear target con cierta relación
        score = 0.4*textura + 0.3*contraste + 0.2*forma_nucleo + 0.001*area_cell + 0.2*densidad
        prob = 1/(1 + np.exp(-5*(score - np.median(score))))
        target = (rng.rand(n) < prob).astype(int)
        df = pd.DataFrame({
            "textura": textura,
            "contraste": contraste,
            "forma_nucleo": forma_nucleo,
            "area_cell": area_cell,
            "densidad": densidad,
            TARGET_NAME: target
        })
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        return df
    else:
        df = pd.read_csv(DATA_PATH)
        # validar que las columnas estén presentes
        missing = [c for c in FEATURE_COLUMNS + [TARGET_NAME] if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en el dataset: {missing}")
        return df

def train_and_save(test_size=0.2, random_state=RANDOM_STATE):
    """
    Entrena un SVM con pipeline (StandardScaler + SVC(probability=True)).
    Guarda modelo con joblib y genera imágenes (matriz confusión y ROC).
    Retorna diccionario con métricas y rutas de imágenes.
    """
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_NAME].astype(int)

    # split 80/20 con estratificación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Pipeline: escalado dentro del pipeline para evitar leakage
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True, random_state=random_state))
    ])

    pipeline.fit(X_train, y_train)

    # Predicción y métricas
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # probabilidad clase positiva
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    cm = confusion_matrix(y_test, y_pred)

    # Crear carpetas
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    # Guardar modelo (pipeline incluye scaler)
    joblib.dump({
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "target_name": TARGET_NAME
    }, MODEL_PATH)

    ts = str(int(time.time()))
    cm_path = os.path.join(IMG_DIR, f"confusion_svm_{ts}.png")
    # Guardar matriz de confusión (2x2)
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Matriz de confusión (test)")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No (0)","Sí (1)"]); ax.set_yticklabels(["No (0)","Sí (1)"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(cm_path)
    plt.close(fig)

    # Opcional: ROC
    roc_path = os.path.join(IMG_DIR, f"roc_svm_{ts}.png")
    try:
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig2, ax2 = plt.subplots(figsize=(5,4))
        ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax2.plot([0,1],[0,1], linestyle="--", color="gray")
        ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.set_title("ROC curve (test)")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(roc_path)
        plt.close(fig2)
    except Exception:
        roc_path = None

    # Preparar report HTML para insertar en template
    report_html = report_df.to_html(classes="table table-striped table-bordered", float_format="%.3f")

    return {
        "accuracy": round(float(acc), 4),
        "report_html": report_html,
        "confusion_img": "/static/images/" + os.path.basename(cm_path),
        "roc_img": ("/static/images/" + os.path.basename(roc_path)) if roc_path else None,
        "model_path": MODEL_PATH
    }

def load_model():
    """Carga el pipeline guardado y devuelve pipeline, columnas"""
    if not os.path.exists(MODEL_PATH):
        # si no existe, entrenar y guardar
        train_and_save()
    payload = joblib.load(MODEL_PATH)
    return payload["pipeline"], payload["feature_columns"]

def evaluate():
    """
    Conveniencia para la interfaz: entrena (si es necesario) y retorna métricas y rutas.
    """
    return train_and_save()

def predict_label(features: dict, threshold: float = 0.5):
    """
    features: dict con keys de FEATURE_COLUMNS (puede pasar valores faltantes -> 0 por defecto)
    retorna: ("Sí"/"No", probabilidad_float)
    """
    pipeline, cols = load_model()
    x = np.array([features.get(c, 0.0) for c in cols]).reshape(1, -1)
    # predict_proba devuelve prob de clase 1 en columna 1
    proba = pipeline.predict_proba(x)[0,1]
    label = "Sí" if proba >= threshold else "No"
    return label, float(proba)

# Si ejecutas directo, imprime métricas
if __name__ == "__main__":
    metrics = evaluate()
    print("Accuracy:", metrics["accuracy"])
    print("Reporte (HTML):", metrics["report_html"][:200])
    print("Confusion image:", metrics["confusion_img"])
