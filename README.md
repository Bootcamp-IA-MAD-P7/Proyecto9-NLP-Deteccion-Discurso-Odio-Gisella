# HateShield — Detección de Discurso de Odio en Comentarios de YouTube

Sistema de clasificación automática de comentarios de YouTube para la detección de
discurso de odio, desarrollado como solución de moderación de contenido a petición
de un cliente que necesita escalar su proceso de moderación.

## Índice

- [Problema de negocio](#problema-de-negocio)
- [Dataset](#dataset)
- [Objetivo del modelo](#objetivo-del-modelo)
- [Stack tecnológico](#stack-tecnológico)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Cómo levantar el entorno](#cómo-levantar-el-entorno)
- [Roadmap por niveles](#roadmap-por-niveles)
- [Flujo de trabajo (Git/Kanban)](#flujo-de-trabajo-gitkanban)
- [Fechas clave](#fechas-clave)
- [Estado actual](#estado-actual)

---

## Problema de negocio

YouTube recibe un volumen muy alto de comentarios en sus vídeos, lo que hace inviable
la moderación manual completa. El objetivo de este proyecto es desarrollar una solución
de Machine Learning que **detecte automáticamente comentarios con discurso de odio**,
para que el equipo de moderación humana concentre sus recursos en los casos señalados
por el modelo.

La prioridad es una **solución práctica y utilizable**, no únicamente la máxima precisión.

## Dataset

- **Fuente:** dataset de comentarios de YouTube proporcionado por el Bootcamp
  (`youtoxic_english_1000.csv`).
- **Registros:** 1000 comentarios, 15 columnas, sin valores nulos.
- **Variable objetivo:** `IsHatespeech` (138 positivos / 862 negativos — dataset
  desbalanceado, se gestionará explícitamente en el preprocesamiento/modelado).
- Se valorará ampliar el dataset (otro dataset público, web scraping o data
  augmentation) únicamente si el EDA demuestra una limitación real del dataset
  original, manteniendo siempre el espíritu del proyecto.

> **Nota:** el dataset no se incluye en este repositorio (no es propiedad del
> proyecto). Para reproducir el análisis, descarga `youtoxic_english_1000.csv`
> y colócalo dentro de la carpeta `data/`.

## Objetivo del modelo

Clasificación binaria: dado un comentario de YouTube, predecir si contiene discurso
de odio (`IsHatespeech = True/False`).

Se compararán varios modelos de clasificación de texto y se seleccionará el que
ofrezca el mejor equilibrio entre rendimiento en la clase minoritaria (odio) y
control del overfitting.

## Stack tecnológico

| Categoría | Tecnología |
|---|---|
| Lenguaje | Python 3.12 |
| Manipulación de datos | Pandas |
| NLP clásico | NLTK / spaCy, regex |
| Vectorización | Bag of Words, TF-IDF |
| Machine Learning | scikit-learn (Regresión Logística, Naive Bayes, SVM, Random Forest) |
| Ensemble | Voting / Stacking Classifier |
| Ajuste de hiperparámetros | GridSearchCV / Optuna |
| Deep Learning (nivel avanzado) | RNN / LSTM |
| Transformers (nivel experto) | Hugging Face |
| Aplicación | Streamlit |
| Base de datos | SQLite (histórico de predicciones) |
| Experiment tracking (nivel experto) | MLflow |
| Contenedorización | Docker |
| Control de versiones | Git / GitHub (GitHub Projects para Kanban) |

*(Esta tabla se irá actualizando a medida que se tomen decisiones técnicas.)*

## Estructura del proyecto

```
Proyecto9-NLP-Deteccion-Discurso-Odio-Gisella/
├── data/                  # Datasets (crudo y procesado)
├── notebooks/             # Notebooks de exploración y experimentación
├── src/                   # Código fuente reutilizable (preprocesamiento, modelo, etc.)
├── app/                   # Aplicación Streamlit
├── tests/                 # Tests unitarios
├── models/                # Modelos entrenados serializados
├── .venv/                 # Entorno virtual (no versionado)
├── requirements.txt       # Dependencias del proyecto
└── README.md
```

## Cómo levantar el entorno

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno (Windows)
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Roadmap por niveles

### 🟢 Nivel esencial
- [ ] Modelo de ML que detecte mensajes de odio
- [ ] Overfitting controlado (diferencia train/test < 5 puntos)
- [ ] Aplicación funcional (Streamlit) para consultar si un mensaje es de odio
- [ ] Repositorio Git organizado, commits limpios y descriptivos
- [ ] Documentación y README

### 🟡 Nivel medio
- [ ] Modelo con técnicas de ensemble
- [ ] Detección a partir de la URL de un vídeo de YouTube
- [ ] Tests unitarios
- [ ] Ajuste de hiperparámetros (Optuna / GridSearch)

### 🟠 Nivel avanzado
- [ ] Modelo de Deep Learning (RNN/LSTM) comparado con ML clásico
- [ ] Seguimiento en tiempo real de un vídeo (polling periódico)
- [ ] Despliegue en servidor público
- [ ] Dockerización

### 🔴 Nivel experto
- [ ] Modelo basado en Transformers
- [ ] Persistencia de predicciones en base de datos (SQLite)
- [ ] Tracking de experimentos con MLflow

## Flujo de trabajo (Git/Kanban)

- **Ramas:** todo el desarrollo ocurre en `developer`; al finalizar y validar, se
  hace merge a `main` para la entrega.
- **Flujo:** Issue → rama de funcionalidad → commit → Pull Request → `developer` → `main`.
- **Idiomas:** código, commits y Pull Requests en inglés; documentación en español.
- **Gestión de tareas:** tablero Kanban en GitHub Projects (Backlog → To Do →
  In Progress → Review → Done).

## Fechas clave

- **Entrega:** 22 de septiembre de 2026
- **Presentación (presencial):** 23 de septiembre de 2026

## Estado actual

- [x] Entorno virtual configurado
- [x] Dataset cargado y explorado (estructura, nulos, duplicados, balance de clases)
- [x] Variable objetivo decidida (`IsHatespeech`)
- [ ] EDA completo (en curso)
- [ ] Preprocesamiento NLP
- [ ] Vectorización
- [ ] Entrenamiento y comparación de modelos
