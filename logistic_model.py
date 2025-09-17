import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dataset_gimnasio.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'gym_model.pkl')
IMG_PATH = os.path.join(BASE_DIR, 'static', 'images', 'confusion_matrix.png')


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    # Convertir Sí/No a 1/0
    if 'uso_clases' in df.columns:
        df['uso_clases'] = df['uso_clases'].map({'Sí': 1, 'Si': 1, 'No': 0}).fillna(0)
    return df


def train_and_save(test_size=0.2, random_state=42):
    """
    Entrena el modelo, guarda el modelo, el scaler y la imagen de la matriz
    de confusión, y retorna un diccionario con las métricas para usarlas en Flask.
    """
    df = load_dataset()
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Métricas
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_html = report_df.to_html(classes="table table-striped table-bordered", float_format="%.2f")
    cm = confusion_matrix(y_test, y_pred)

    # Guardar imagen de matriz de confusión
    os.makedirs(os.path.dirname(IMG_PATH), exist_ok=True)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    ax.set_title('Matriz de confusión')
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['0', '1'])
    ax.set_yticklabels(['0', '1'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(IMG_PATH)
    plt.close(fig)

    # Guardar modelo y scaler
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler, 'columns': X.columns.tolist()}, MODEL_PATH)

    # Retornar métricas para la interfaz web
    return {
        "accuracy": round(acc, 4),
        "report": report_html,
        "confusion_matrix": cm.tolist(),
        "model_path": MODEL_PATH,
        "confusion_img": "/static/images/confusion_matrix.png"  # <-- ruta pública
    }


def load_model():
    payload = joblib.load(MODEL_PATH)
    return payload['model'], payload['scaler'], payload['columns']


def predict_label(features: dict, threshold: float = 0.5):
    model, scaler, cols = load_model()
    x = np.array([features.get(c, 0) for c in cols]).reshape(1, -1)
    x_scaled = scaler.transform(x)
    prob = model.predict_proba(x_scaled)[0, 1]
    return ('Sí' if prob >= threshold else 'No', float(prob))


if __name__ == '__main__':
    metrics = train_and_save()
    # Si se ejecuta directo, mostramos las métricas por consola
    print("Accuracy:", metrics["accuracy"])
    print("Reporte de clasificación:", metrics["report"])
    print("Matriz de confusión:", metrics["confusion_matrix"])
