from flask import Flask, render_template, abort, request, url_for
from regresion_lineal import predict, plot_data_and_regression, plot_new_data  # funciones reales
import os

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

# ===== Referencias (divididas por tema) =====
referencias_supervisado = [
    '<a href="https://journalretinavitreous.biomedcentral.com/articles/10.1186/s40942-021-00352-2" target="_blank">Detección de retinopatía diabética</a>',
    '<a href="https://breast-cancer-research.biomedcentral.com/articles/10.1186/s13058-024-01895-6" target="_blank">Detección de cáncer de mama en mamografías</a>',
    '<a href="https://seon.io/es/recursos/machine-learning-para-detectar-fraude/" target="_blank">Detección de fraude en pagos</a>',
    '<a href="https://www.advancedtech.com/blog/machine-learning-predictive-maintenance/" target="_blank">Mantenimiento predictivo</a>'
]

referencias_regresion = [
    '<a href="https://scikit-learn.org/stable/modules/linear_model.html" target="_blank">Documentación oficial de scikit-learn: Linear Regression</a>',
    '<a href="https://towardsdatascience.com/a-complete-guide-to-linear-regression-in-python-83c2f1282f1c" target="_blank">Guía práctica de regresión lineal en Python</a>',
    '<a href="https://statisticsbyjim.com/regression/linear-regression-tutorial/" target="_blank">Tutorial sobre regresión lineal</a>'
]

# ===== Rutas Casos =====
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
    return render_template(
        'referencias.html',
        referencias_supervisado=referencias_supervisado,
        referencias_regresion=referencias_regresion,
        cases=cases
    )

# ===== Rutas Regresión Lineal =====
@app.route('/regresion/conceptos')
def regresion_conceptos():
    return render_template(
        'regresion_conceptos.html',
        referencias=referencias_regresion,
        cases=cases
    )

@app.route('/regresion/practico', methods=['GET', 'POST'])
def regresion_practico():
    prediction = None
    user_plot = None  # usamos el mismo nombre que en el template
    if request.method == 'POST':
        estatura = float(request.form['estatura'])
        edad = float(request.form['edad'])
        prediction = predict(estatura, edad)

        # Gráfico con el punto ingresado por el usuario
        user_plot = plot_new_data(estatura, edad, prediction)

    # Gráficos de entrenamiento
    plot1, plot2 = plot_data_and_regression()

    return render_template(
        'regresion_practico.html',
        prediction=prediction,
        plot1=plot1,
        plot2=plot2,
        user_plot=user_plot,   
        cases=cases
    )
# ===== Rutas Regresión Logística =====
@app.route('/regresion_logistica/conceptos')
def regresion_logistica_conceptos():
    return render_template(
        'regresion_logistica_conceptos.html',
        cases=cases
    )

@app.route('/regresion_logistica/practico')
def regresion_logistica_practico():
    return render_template(
        'regresion_logistica_practico.html',
        cases=cases
    )

if __name__ == '__main__':
    app.run(debug=True)
