# regresion_lineal.py
import os
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump, load
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_DIR = os.path.join(BASE_DIR, "static", "images")

DATA_PATH = os.path.join(DATA_DIR, "weight_height_age.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "linreg_model.joblib")

def generate_sample_dataset(n=200, seed=42):
    rng = np.random.RandomState(seed)
    heights = rng.normal(loc=1.70, scale=0.08, size=n)  # metros
    ages = rng.normal(loc=35, scale=12, size=n).clip(5, 90).round(0)
    weight = 50 + 20 * (heights - 1.5) + 0.2 * ages + rng.normal(scale=3.0, size=n)
    df = pd.DataFrame({"height_m": heights, "age_yr": ages, "weight_kg": weight})
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df

def load_dataset():
    if not os.path.exists(DATA_PATH):
        return generate_sample_dataset()
    return pd.read_csv(DATA_PATH)

def train_and_save_model():
    df = load_dataset()
    X = df[["height_m", "age_yr"]]
    y = df["weight_kg"]
    model = LinearRegression()
    model.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(model, MODEL_PATH)
    return model, df

def load_model():
    if not os.path.exists(MODEL_PATH):
        model, _ = train_and_save_model()
        return model
    return load(MODEL_PATH)

def predict(height_m, age_yr):
    model = load_model()
    Xnew = [[float(height_m), float(age_yr)]]
    pred = model.predict(Xnew)[0]
    return float(pred)

def plot_data_and_regression():
    """
    Genera dos gráficos y los guarda en static/images:
     - peso vs estatura (edad color)
     - peso vs edad (estatura color)
    Devuelve (ruta_relativa1, ruta_relativa2) aptas para usar en templates con url_for('static', ...)
    """
    df = load_dataset()
    model = load_model()
    os.makedirs(IMG_DIR, exist_ok=True)
    ts = str(int(time.time()))
    file1 = os.path.join(IMG_DIR, f"reg_plot_height_{ts}.png")
    file2 = os.path.join(IMG_DIR, f"reg_plot_age_{ts}.png")

    # Plot 1: Peso vs Estatura (edad promedio para la línea)
    mean_age = df["age_yr"].mean()
    heights = np.linspace(df["height_m"].min(), df["height_m"].max(), 100)
    X_pred = pd.DataFrame({"height_m": heights, "age_yr": mean_age})
    y_pred = model.predict(X_pred)

    fig, ax = plt.subplots(figsize=(7,4))
    sc = ax.scatter(df["height_m"], df["weight_kg"], c=df["age_yr"], cmap="viridis", alpha=0.7)
    ax.plot(heights, y_pred, color="red", linewidth=2, label=f"Edad media ≈ {mean_age:.1f}")
    ax.set_xlabel("Estatura (m)")
    ax.set_ylabel("Peso (kg)")
    ax.set_title("Peso vs Estatura (puntos coloreados por edad)")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="Edad (años)")
    fig.tight_layout()
    fig.savefig(file1)
    plt.close(fig)

    # Plot 2: Peso vs Edad (estatura promedio para la línea)
    mean_height = df["height_m"].mean()
    ages = np.linspace(df["age_yr"].min(), df["age_yr"].max(), 100)
    X_pred2 = pd.DataFrame({"height_m": mean_height, "age_yr": ages})
    y_pred2 = model.predict(X_pred2)

    fig2, ax2 = plt.subplots(figsize=(7,4))
    sc2 = ax2.scatter(df["age_yr"], df["weight_kg"], c=df["height_m"], cmap="plasma", alpha=0.7)
    ax2.plot(ages, y_pred2, color="red", linewidth=2, label=f"Estatura media ≈ {mean_height:.2f} m")
    ax2.set_xlabel("Edad (años)")
    ax2.set_ylabel("Peso (kg)")
    ax2.set_title("Peso vs Edad (puntos coloreados por estatura)")
    ax2.legend()
    fig2.colorbar(sc2, ax=ax2, label="Estatura (m)")
    fig2.tight_layout()
    fig2.savefig(file2)
    plt.close(fig2)

    # Rutas relativas para usar con url_for('static', filename='images/xxx.png')
    rel1 = "images/" + os.path.basename(file1)
    rel2 = "images/" + os.path.basename(file2)
    return rel1, rel2

# Si ejecutas directamente, entrena y genera un par de plots
if __name__ == "__main__":
    train_and_save_model()
    print("Modelo entrenado y guardado en:", MODEL_PATH)
    p1, p2 = plot_data_and_regression()
    print("Plots generados:", p1, p2)
