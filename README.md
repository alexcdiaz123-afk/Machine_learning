# Machine_learning

# DOCUMENTO CREADO PARA VER EL AVANCE DEL PROYECTO Y COMO SE ESTA ESTRUCTURANDO 
# Proyecto: #1 Casos de Uso de Machine Learning Supervisado con Flask

Este proyecto es una aplicación web desarrollada con Flask que presenta cuatro casos de uso relevantes del Machine Learning supervisado en distintas industrias, mostrando cómo se aplican algoritmos para resolver problemas reales.

# Funcionalidades

Página de inicio: Introducción a los casos de uso investigados.

Sección de casos de éxito: Cada caso incluye:

El problema que se resolvió.

El tipo de algoritmo supervisado utilizado.

Los beneficios obtenidos.

Ejemplos concretos de empresas o proyectos que implementaron la solución.

Referencias: Enlaces a los recursos consultados para cada caso de uso.

# Casos de uso incluidos

Detección de retinopatía diabética – Salud: diagnóstico temprano mediante imágenes de retina.

Detección de cáncer de mama en mamografías – Salud: reducción de falsos positivos/negativos en radiología.

Detección de fraude en pagos – Finanzas: identificación de transacciones fraudulentas en tiempo real.

Mantenimiento predictivo – Industria: predicción de fallos en equipos para optimizar mantenimiento.

# Tecnologías y herramientas

Flask: Framework web para Python, usado para crear rutas y servir las páginas.

Jinja2: Sistema de templates para renderizar dinámicamente los datos en HTML.

Python: Lenguaje principal del proyecto.

# Uso de funciones de Flask

url_for: Genera URLs dinámicas para los links de navegación y botones, asegurando que los enlaces funcionen correctamente incluso si cambian las rutas.

abort: Permite manejar errores y mostrar una página 404 cuando se intenta acceder a un caso que no existe en la base de datos de la aplicación.

# Uso de IA como apoyo

Durante el desarrollo del proyecto se utilizó esta IA (ChatGPT y Gemini) como herramienta de apoyo para comprender mejor cómo funciona el código de Flask, especialmente en:

La explicación de url_for, que fue clave para entender cómo generar rutas dinámicas de forma segura.

El uso de abort, que permite manejar casos de error y mejorar la robustez de la aplicación.

Resolver dudas conceptuales sobre la estructura de Flask y la interacción con los templates Jinja2.

El uso de la IA fue puramente explicativo y de apoyo académico, complementando la investigación y la implementación realizada en el proyecto.

# Estructura del proyecto

app.py: Contiene la lógica de Flask, rutas y los datos de los casos de uso.

templates/: Carpeta con los templates HTML (index.html, case.html, referencias.html, base.html).

static/: Carpeta con archivos de estilo y otros recursos estáticos (por ejemplo styles.css).

# Proyecto: #2 Regresiones Lineales:

Este módulo del proyecto integra la implementación de una regresión lineal aplicada a la predicción del peso de una persona según su estatura y edad, mostrando de manera práctica cómo se entrenan y visualizan los modelos supervisados.

# Funcionalidades:

se agregaron dos nuevas funcionalidades al proyecto

# Conceptos básicos:

Se incluyen explicaciones teóricas sobre la regresión lineal, sus fórmulas y referencias académicas para entender su funcionamiento.

# Ejercicio práctico:
Permite visualizar los datos de entrenamiento, la línea de regresión ajustada y realizar predicciones con nuevos datos.

# Uso de IA como apoyo

Durante el desarrollo del proyecto se utilizaron herramientas de inteligencia artificial, como ChatGPT y Gemini, como apoyo para comprender y aplicar conceptos relacionados con regresiones en Python.

Se utilizó la IA para entender el funcionamiento de las regresiones lineales y otras técnicas de predicción, facilitando la interpretación de resultados y su aplicación correcta en el proyecto.

Se consultó código de ejemplo proporcionado por la IA, lo que ayudó a aprender buenas prácticas y adaptar soluciones a las necesidades específicas del proyecto.

La IA también sirvió para resolver problemas técnicos y errores comunes, como dificultades en la instalación de paquetes o dependencias definidas en el requirements.txt, asegurando que el entorno de desarrollo estuviera correctamente configurado.

El uso de la IA fue puramente explicativo y de apoyo académico, complementando la investigación y la implementación realizada por el equipo.
# Proyecto: #3 Regresiones Logísticas

Este módulo del proyecto integra la implementación de una regresión logística aplicada a la predicción de deserción en un gimnasio. 
La regresión logística es un algoritmo supervisado utilizado principalmente para problemas de clasificación binaria, donde el objetivo es predecir 
si un evento ocurrirá o no (ejemplo: “sí/no”, “0/1”, “aprobado/rechazado”).

# Funcionalidades:

Se agregaron dos nuevas funcionalidades al proyecto:

- **Conceptos básicos:** explicación teórica sobre la regresión logística, el uso de la función sigmoide/logit, la variable objetivo binaria y ejemplos prácticos de aplicación en diferentes contextos.
- **Ejercicio práctico:** implementación de un modelo de clasificación que predice si un usuario se dará de baja en un gimnasio a partir de variables relacionadas con su comportamiento.

# Conceptos básicos:

La regresión logística se centra en problemas de clasificación binaria. A través de la función sigmoide, convierte las entradas en probabilidades entre 0 y 1, permitiendo decidir si un caso pertenece a la clase positiva (“1”) o negativa (“0”).  
También se utiliza el **logit**, que transforma la razón de probabilidades (odds) en una escala lineal para relacionar predictores con la probabilidad de ocurrencia del evento.

En este proyecto se incluyen explicaciones breves y referencias académicas que permiten entender el funcionamiento del modelo y sus principales aplicaciones.

# Ejercicio práctico:

Se desarrolló un ejercicio práctico de predicción de deserción en un gimnasio, con el siguiente planteamiento:

- **Variables predictoras:**  
  - Frecuencia de visitas  
  - Tiempo inscrito  
  - Uso de clases grupales (categórica: Sí/No)  
  - Edad  

- **Variable objetivo:**  
  - ¿Se da de baja? (Sí/No)  

El código implementado (`logistic_model.py`) permite:  
- Entrenar el modelo de regresión logística.  
- Escalar y preparar los datos para un mejor ajuste.  
- Guardar el modelo entrenado y su scaler para realizar predicciones en la interfaz web.  
- Generar métricas de evaluación como accuracy, reporte de clasificación y matriz de confusión.  
- Guardar una visualización de la matriz de confusión para analizar el desempeño del modelo.  

# Uso de IA como apoyo

Durante el desarrollo del módulo de regresión logística, se utilizaron herramientas de inteligencia artificial como ChatGPT y Gemini como apoyo en:

- Comprender el funcionamiento de la regresión logística, la función sigmoide y el logit.  
- Adaptar código de ejemplo en Python a las necesidades específicas del proyecto.  
- Resolver dudas conceptuales sobre la clasificación binaria y las métricas de evaluación.  
- Solucionar errores técnicos durante la implementación, como el manejo de variables categóricas, la preparación del dataset y la generación de gráficos.  

El uso de la IA fue principalmente de apoyo académico y técnico, asegurando una mejor comprensión y correcta implementación del modelo en el proyecto.
# Proyecto: Algoritmos de Clasificación – Support Vector Machines (SVM):
Este módulo del proyecto integra la implementación de un algoritmo de clasificación (SVM) dentro de la aplicación web Flask ya creada.
El objetivo es predecir si una célula es Cancerosa o No cancerosa, a partir de características extraídas de imágenes celulares.

# Funcionalidades

Se agregaron nuevas funcionalidades al proyecto:

Pestaña en la web: Tipos de Algoritmos de Clasificación.

Submenús:

Conceptos básicos: síntesis teórica del algoritmo, presentada en un mapa conceptual en MindMeister con referencias APA 7.

Caso práctico (SVM – Imágenes celulares):

Carga de datos.

Entrenamiento y evaluación del modelo con métricas.

Predicción Sí/No en la interfaz web.

Desarrollo técnico:

Creación de un script .py con el modelo de SVM.

Implementación de funciones estándar para todos los algoritmos:

evaluate() → retorna métricas y genera matriz de confusión.

predict_label(features, threshold=0.5) → retorna “Sí/No” y la probabilidad asociada.

Integración con Flask:

Nuevas rutas en app.py para renderizar el formulario y mostrar resultados.

Creación de un archivo HTML específico para el práctico, con formulario y visualización de resultados.

# Conceptos básicos

El algoritmo Support Vector Machines (SVM) se utiliza para clasificación binaria y multiclase. Su objetivo es encontrar un hiperplano que separe los datos de manera óptima, maximizando el margen entre clases.

En este proyecto:

Se trabaja con un caso binario: Célula cancerosa (1) vs No cancerosa (0).

Se aplicó escalado de variables con StandardScaler dentro de un Pipeline, evitando data leakage.

Se documentó el significado de las clases y variables:

Variables independientes:

Textura de la imagen

Contraste

Forma del núcleo

Área celular

Densidad

Variable objetivo:

Tipo de célula → Cancerosa (1) / No cancerosa (0)

# Ejercicio práctico: Clasificación de imágenes celulares

El caso práctico implementado incluye:

Carga de datos y preprocesamiento:

División 80/20 entre entrenamiento y prueba.

Escalado de variables numéricas.

Entrenamiento del modelo:

Algoritmo asignado: SVM.

Control de semilla para reproducibilidad.

Evaluación en test:

Exactitud (accuracy) mostrada en una tarjeta.

Reporte de clasificación (precision, recall, F1, support).

Matriz de confusión 2×2 con etiquetas claras (Real vs Predicho).

Formulario de predicción en Flask:

Inputs numéricos para cada variable independiente.

Campo opcional de umbral (threshold).

Botón “Predecir”.

Resultado dinámico:

Texto grande: “Predicción: Sí” o “Predicción: No”.

Probabilidad entre [0,1] (4 decimales).

Interpretación breve sobre el umbral aplicado.

# Uso de IA como apoyo

Durante el desarrollo del caso práctico, se utilizó ChatGPT como apoyo en:
Generar las funciones evaluate() y predict_label() con buenas prácticas.

Resolver problemas de instalación de dependencias en Python 3.13 y configurar un entorno virtual (.venv) con requirements.txt

El uso de IA fue de apoyo académico y técnico, facilitando la implementación del modelo de clasificación en la aplicación web.
Proyecto: #5 Aprendizaje por Refuerzo – Q-Learning (Supermercado Inteligente)

# Reinforcement Learning:

Este módulo del proyecto integra la implementación de un algoritmo de Aprendizaje por Refuerzo (Q-Learning) dentro de la aplicación web Flask. El objetivo es simular cómo un agente (un robot) aprende a desplazarse dentro de un entorno tipo supermercado, tomando decisiones óptimas mediante prueba y error hasta aprender la mejor ruta hacia su objetivo.

# Funcionalidades

Se agregaron nuevas funcionalidades al proyecto:

Pestaña en la interfaz web dedicada al caso práctico de Aprendizaje por Refuerzo.

Formulario interactivo que permite configurar:

Número de episodios.

Tasa de aprendizaje (alpha).

Factor de descuento (gamma).

Nivel de exploración (epsilon).

Tamaño del mapa.

Entrenamiento del agente Q-Learning desde el navegador, con generación automática de:

Gráfica de recompensas obtenidas durante el entrenamiento.

Representación visual del recorrido óptimo aprendido.

Simulación del comportamiento del agente utilizando la tabla Q ya entrenada.

# Descripción del entorno

El entorno es una cuadrícula (grid) que simula un supermercado. En ella:

El robot puede moverse arriba, abajo, izquierda o derecha.

Cada movimiento otorga una recompensa:

Penalización por movimientos irrelevantes o alejarse del objetivo.

Recompensa positiva al llegar a la meta.

Un episodio termina cuando el robot alcanza el objetivo o se excede el número máximo de pasos.

La simulación final genera una imagen del camino que el agente aprendió como óptimo.

Este entorno facilita comprender cómo un agente aprende mediante interacción directa, sin supervisión humana.

# Algoritmo utilizado

El modelo usa Q-Learning, uno de los algoritmos fundamentales del Aprendizaje por Refuerzo. Su objetivo es aprender una tabla Q que almacena el valor esperado de tomar cada acción en cada estado.

El proceso implementado incluye:

Política epsilon-greedy para equilibrar exploración y explotación.

Actualización de valores Q mediante la ecuación de Bellman.

Entrenamiento por episodios hasta estabilizar la política aprendida.

Simulación posterior con política completamente explotativa (sin exploración).

La implementación se encuentra en el archivo rl_agent.py.

Comportamiento obtenido

Durante el entrenamiento se observa:

Que el agente inicia moviéndose al azar y recibiendo recompensas bajas.

Que progresivamente encuentra rutas más cortas y consistentes.

Que la recompensa promedio aumenta conforme mejora la política.

Que la simulación final muestra el recorrido óptimo aprendido según los valores Q entrenados.

El sistema genera dos gráficos:

Evolución de recompensas durante el entrenamiento.

Recorrido óptimo del agente en el entorno.

# Integración con Flask

Para conectar el algoritmo con la interfaz, se añadieron dos rutas en app.py:

/train (POST): ejecuta el entrenamiento con los parámetros ingresados por el usuario.

/simulate (GET): ejecuta la simulación final con la tabla Q ya entrenada.

El archivo rl_practico.html maneja la interacción mediante JavaScript, mostrando en la página los resultados, métricas y gráficas generadas.

# Uso de IA como apoyo

Durante el desarrollo del módulo de Aprendizaje por Refuerzo se utilizaron herramientas de inteligencia artificial como ChatGPT como apoyo para:

Comprender la estructura del algoritmo Q-Learning y su correcta implementación.

Resolver dudas sobre el diseño del entorno y la forma de otorgar recompensas.

Integrar el entrenamiento y la simulación con rutas Flask y peticiones fetch.

Depurar errores relacionados con el intercambio de datos JSON y la generación de imágenes desde Python.

Aclarar el funcionamiento de la política epsilon-greedy y la actualización de la tabla Q.

El uso de la IA fue de apoyo académico y técnico, complementando la investigación conceptual y el desarrollo realizado en el proyecto.