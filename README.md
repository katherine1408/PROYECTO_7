# 🎮 Sprint 7 – Análisis Integrado de Datos: Ventas de Videojuegos (ICE Store)

## 📌 Descripción del Proyecto

Este proyecto integra habilidades previas de manipulación, visualización y análisis estadístico para abordar un caso real: **la planificación de campañas publicitarias basadas en datos históricos de ventas de videojuegos**.

Trabajamos para **ICE**, una tienda online que vende videojuegos en todo el mundo. Se analizarán plataformas, géneros, calificaciones y reseñas para identificar patrones que indiquen el éxito de un videojuego.

El análisis se basa en un dataset que contiene información de videojuegos hasta 2016. El objetivo es tomar decisiones para el año 2017.

## 🎯 Propósito

- Determinar factores clave de éxito en videojuegos: plataforma, género, puntuación, región.
- Identificar patrones históricos de ventas por plataforma y región.
- Analizar impacto de calificaciones y reseñas en las ventas.
- Comparar comportamiento del consumidor en Norteamérica, Europa y Japón.
- Aplicar pruebas estadísticas a hipótesis de comparación entre plataformas y géneros.

## 🧰 Funcionalidad del Proyecto

### 🧹 Limpieza y preprocesamiento
- Conversión de nombres de columnas a minúsculas.
- Conversión de tipos de datos.
- Tratamiento de valores nulos y casos "TBD".
- Cálculo de una columna de ventas globales (`total_sales`).

### 📈 Análisis exploratorio
- Tendencias de ventas por plataforma y año.
- Identificación de plataformas líderes y emergentes.
- Comparación de reseñas de usuarios y críticos con ventas.
- Análisis de ventas por género y ESRB rating.

### 🌍 Perfil de usuario por región
- Top 5 plataformas y géneros por región: NA, EU, JP.
- Análisis de impacto del rating ESRB por región.

### 🔬 Pruebas de hipótesis
- Comparación de calificaciones entre Xbox One y PC.
- Diferencia de calificaciones entre géneros Acción y Deportes.
- Uso de pruebas estadísticas con α definido.

## 📁 Archivo utilizado
- `games.csv`

## 📊 Herramientas utilizadas

- Python  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scipy.stats  

---

📌 Proyecto desarrollado como parte del Sprint 7 del programa de Ciencia de Datos en **TripleTen**.
