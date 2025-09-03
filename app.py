from flask import Flask, render_template, abort, url_for

app = Flask(__name__)

# ===== Datos de investigación (casos de uso) =====
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

# ===== Referencias en formato HTML clicable =====
referencias_apa = [
    '<a href="https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi79riDk7uPAxUNRzABHd60MOAQFnoECB4QAQ&url=https%3A%2F%2Ftranslate.google.com%2Ftranslate%3Fu%3Dhttps%3A%2F%2Fjournalretinavitreous.biomedcentral.com%2Farticles%2F10.1186%2Fs40942-021-00352-2%26hl%3Des%26sl%3Den%26tl%3Des%26client%3Dsrp&usg=AOvVaw2qeTyXbZcHWef8TTDQw_ee&opi=89978449" target="_blank">Detección de retinopatía diabética</a>',
    '<a href="https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjL4PKWlLuPAxUXRDABHWCSAv0QFnoECBgQAQ&url=https%3A%2F%2Ftranslate.google.com%2Ftranslate%3Fu%3Dhttps%3A%2F%2Fbreast-cancer-research.biomedcentral.com%2Farticles%2F10.1186%2Fs13058-024-01895-6%26hl%3Des%26sl%3Den%26tl%3Des%26client%3Dsrp&usg=AOvVaw10gEgPuJPE8iEiznY3KlHA&opi=89978449" target="_blank">Detección de cáncer de mama en mamografías</a>',
    '<a href="https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiGndTRlLuPAxURSTABHYYyBOsQFnoECB8QAQ&url=https%3A%2F%2Fseon.io%2Fes%2Frecursos%2Fmachine-learning-para-detectar-fraude%2F&usg=AOvVaw0CRIXu_Q1UFoM5VFPsGtsY&opi=89978449" target="_blank">Detección de fraude en pagos</a>',
    '<a href="https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwj2lsSJlbuPAxXtQTABHapsBXkQFnoECB4QAQ&url=https%3A%2F%2Ftranslate.google.com%2Ftranslate%3Fu%3Dhttps%3A%2F%2Fwww.advancedtech.com%2Fblog%2Fmachine-learning-predictive-maintenance%2F%26hl%3Des%26sl%3Den%26tl%3Des%26client%3Dsrp&usg=AOvVaw3cxD1DaiYxzKO58TZX6_jg&opi=89978449" target="_blank">Mantenimiento predictivo</a>'
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