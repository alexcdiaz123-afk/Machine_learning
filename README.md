# Machine_learning

# Proyecto: Casos de Uso de Machine Learning Supervisado con Flask

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