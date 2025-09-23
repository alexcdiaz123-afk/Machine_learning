from flask import Flask, render_template, abort, request, url_for
from regresion_lineal import predict, plot_data_and_regression, plot_new_data  # funciones reales
import os
from svm_model import evaluate as svm_evaluate, predict_label as svm_predict_label

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
from logistic_model import train_and_save, predict_label   # <--- importa las funciones

@app.route('/regresion_logistica/conceptos')
def regresion_logistica_conceptos():
    return render_template(
        'regresion_logistica_conceptos.html',
        cases=cases
    )

@app.route('/regresion_logistica/practico', methods=['GET', 'POST'])
def regresion_logistica_practico():
    metrics = None
    prediction = None

    if request.method == 'POST':
        # Si el usuario presiona el botón de entrenar
        if 'train' in request.form:
            metrics = train_and_save()

        # Si el usuario envía datos para predecir
        if 'predict' in request.form:
            features = {
                'frecuencia_visitas': float(request.form['frecuencia_visitas']),
                'tiempo_inscrito': float(request.form['tiempo_inscrito']),
                'uso_clases': 1 if request.form['uso_clases'] == 'Sí' else 0,
                'edad': float(request.form['edad'])
            }
            prediction = predict_label(features)

    return render_template(
        'regresion_logistica_practico.html',
        cases=cases,
        metrics=metrics,
        prediction=prediction
    )
    
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
from logistic_model import train_and_save, predict_label   # <--- importa las funciones

@app.route('/regresion_logistica/conceptos')
def regresion_logistica_conceptos():
    return render_template(
        'regresion_logistica_conceptos.html',
        cases=cases
    )

@app.route('/regresion_logistica/practico', methods=['GET', 'POST'])
def regresion_logistica_practico():
    prediction_text = None
    metrics = None
    report_html = None
    cm_img = None

    if request.method == 'POST':
        action = request.form.get('action')

        # Entrenar modelo
        if action == 'train':
            result = train_and_save()  # Retorna métricas, report, confusion_img
            if result:
                metrics = result
                report_html = result.get('report')        # HTML del reporte
                cm_img = result.get('confusion_img')      # Imagen de matriz de confusión

        # Predecir usando datos del formulario
        elif action == 'predict':
            # Asegurarse de que las variables están correctamente indentadas dentro del bloque
            frecuencia = float(request.form['frecuencia_visitas'])
            tiempo = float(request.form['tiempo_inscrito'])
            uso = 1 if request.form['uso_clases'] == 'Sí' else 0
            edad = float(request.form['edad'])
            threshold = float(request.form['threshold'])

            pred = predict_label({
                "frecuencia_visitas": frecuencia,
                "tiempo_inscrito": tiempo,
                "uso_clases": uso,
                "edad": edad
            }, threshold)

            prediction_text = 'Se dará de baja' if pred[0] == 'Sí' else 'Permanecerá'

    # Renderizar template
    return render_template(
        'regresion_logistica_practico.html',
        prediction_text=prediction_text,
        metrics=metrics,
        report_html=report_html,
        cm_img=cm_img
    )
@app.route('/clasificacion/practico', methods=['GET', 'POST'])
def clasificacion_practico():
    # Caso práctico SVM: evaluación y predicción
    metrics = svm_evaluate()  # entrena si es necesario y devuelve métricas
    prediction = None
    prob = None
    threshold = 0.5
    if request.method == 'POST':
        # extraer inputs del form (nombres deben coincidir)
        features = {
            "textura": float(request.form.get("textura", 0)),
            "contraste": float(request.form.get("contraste", 0)),
            "forma_nucleo": float(request.form.get("forma_nucleo", 0)),
            "area_cell": float(request.form.get("area_cell", 0)),
            "densidad": float(request.form.get("densidad", 0))
        }
        # threshold opcional
        try:
            threshold = float(request.form.get("threshold", 0.5))
        except ValueError:
            threshold = 0.5
        prediction, prob = svm_predict_label(features, threshold=threshold)

    return render_template(
        'clasificacion_practico.html',
        metrics=metrics,
        prediction=prediction,
        prob=prob,
        threshold=threshold,
        cases=cases
    )

