from flask import Flask, render_template, abort, url_for

app = Flask(__name__)

# ===== Datos de investigación (puedes editar los textos luego) =====
cases = [
    {
        "id": "retinopatia",
        "titulo": "Salud: Detección de retinopatía diabética",
        "problema": "Detectar retinopatía diabética en imágenes de retina para diagnóstico temprano.",
        "algoritmo": "Redes Neuronales Convolucionales (CNN) para clasificación de imágenes.",
        "beneficios": "Cribado rápido, apoyo a toma de decisiones, mayor cobertura de pacientes.",
        "ejemplo": "IDx-DR (LumineticsCore) aprobado por la FDA para diagnóstico asistido."
    },
    {
        "id": "mamografias",
        "titulo": "Salud: Detección de cáncer de mama en mamografías",
        "problema": "Reducir falsos positivos/negativos en la lectura de mamografías.",
        "algoritmo": "Deep Learning (CNN) entrenadas con grandes conjuntos de mamografías.",
        "beneficios": "Mejor sensibilidad/especificidad y soporte a radiólogos.",
        "ejemplo": "Modelos de Google Health publicados en revistas científicas."
    },
    {
        "id": "fraude",
        "titulo": "Finanzas: Detección de fraude en pagos",
        "problema": "Identificar transacciones fraudulentas en tiempo real.",
        "algoritmo": "Ensembles supervisados (árboles de decisión, bosque aleatorio, gradient boosting).",
        "beneficios": "Menos pérdidas y mejor experiencia de cliente.",
        "ejemplo": "Plataformas como Stripe Radar o Visa Advanced Authorization."
    },
    {
        "id": "mantenimiento",
        "titulo": "Industria: Mantenimiento predictivo",
        "problema": "Predecir fallos de equipos para planificar mantenimiento.",
        "algoritmo": "Modelos supervisados (XGBoost, SVM) sobre datos de sensores.",
        "beneficios": "Reducción de paradas no planificadas y costos.",
        "ejemplo": "Gemelos digitales y analítica en grandes fabricantes (p.ej., GE)."
    }
]

# Referencias en formato APA (reemplaza por tus citas definitivas)
referencias_apa = [
    "Autor, A. A. (Año). Título del artículo. Revista, volumen(número), páginas. https://doi.org/xx",
    "Autor, B. B., & Autor, C. C. (Año). Título del recurso. Editorial.",
    "Organización. (Año). Título del informe. URL",
    "Autor, D. D. (Año). Título del paper relacionado. Conferencia/Revista."
]

# ===== Rutas =====
@app.route('/')
def home():
    return render_template('index.html', cases=cases)

@app.route('/case/<case_id>')
def case_detail(case_id):
    case = next((c for c in cases if c['id'] == case_id), None)
    if not case:
        abort(404)
    return render_template('case.html', case=case, cases=cases)

@app.route('/referencias')
def referencias():
    return render_template('referencias.html', referencias=referencias_apa, cases=cases)

if __name__ == '__main__':
    app.run(debug=True)
