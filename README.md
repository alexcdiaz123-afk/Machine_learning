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
