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
- [Pipeline y resultados](#pipeline-y-resultados)
- [Roadmap por niveles](#roadmap-por-niveles)
- [Flujo de trabajo (Git/Kanban)](#flujo-de-trabajo-gitkanban)
- [Fechas clave](#fechas-clave)
- [Estado actual](#estado-actual)

---

## Problema de negocio

YouTube recibe un volumen muy alto de comentarios en sus vídeos, lo que hace inviable
la moderación manual completa. El objetivo de este proyecto es desarrollar una solución
de Machine Learning que **detecte automáticamente comentarios tóxicos**, para que el
equipo de moderación humana concentre sus recursos en los casos señalados por el modelo.

La prioridad es una **solución práctica y utilizable**, no únicamente la máxima precisión.

## Dataset

- **Fuente:** dataset de comentarios de YouTube proporcionado por el Bootcamp
  (`youtoxic_english_1000.csv`).
- **Registros:** 1000 comentarios originales → 997 tras eliminar 3 duplicados exactos
  por texto. Sin valores nulos ni comentarios vacíos.
- **Variable objetivo:** `IsToxic` (459 positivos / 538 negativos → 46.0% / 54.0%,
  dataset razonablemente balanceado). Se descartó `IsHatespeech` por recomendación
  del profesor, al ser una categoría más específica y menos representada dentro del
  dataset; `IsToxic` ofrece una señal más completa para el problema de negocio.
- Se valoró ampliar el dataset (otro dataset público, web scraping o data
  augmentation), pero el EDA no mostró una limitación real que lo justificara, por lo
  que se mantiene el dataset original.

> **Nota:** el dataset no se incluye en este repositorio. Para reproducir el análisis,
> descarga `youtoxic_english_1000.csv` y colócalo dentro de la carpeta `data/`.

## Objetivo del modelo

Clasificación binaria: dado un comentario de YouTube, predecir si es tóxico
(`IsToxic = True/False`).

Se compararon varios modelos de clasificación de texto y se seleccionó el que ofreció
el mejor equilibrio entre rendimiento en la clase tóxica y control del overfitting.

## Stack tecnológico

| Categoría                           | Tecnología                                                          |
| ------------------------------------ | -------------------------------------------------------------------- |
| Lenguaje                            | Python 3.12                                                         |
| Manipulación de datos               | Pandas                                                              |
| NLP clásico                         | NLTK, regex                                                         |
| Vectorización                       | TF-IDF (unigramas + bigramas)                                       |
| Machine Learning                    | scikit-learn (Regresión Logística, Naive Bayes, SVM, Random Forest) |
| Ajuste de hiperparámetros           | GridSearchCV                                                        |
| Ensemble                            | Random Forest (modelo final)                                        |
| Deep Learning (nivel avanzado)      | RNN / LSTM — pendiente                                              |
| Transformers (nivel experto)        | Hugging Face — pendiente                                            |
| Aplicación                          | Streamlit                                                           |
| Base de datos                       | SQLite (histórico de predicciones) — pendiente                      |
| Experiment tracking (nivel experto) | MLflow — pendiente                                                  |
| Contenedorización                   | Docker — pendiente                                                  |
| Control de versiones                | Git / GitHub (GitHub Projects para Kanban)                          |

## Estructura del proyecto

```
Proyecto9-NLP-Deteccion-Discurso-Odio-Gisella/
├── data/                  # Datasets (no versionado, ver .gitignore)
├── notebooks/             # Notebooks de exploración y experimentación
│   ├── 01_exploracion_dataset.ipynb
│   ├── 02_preprocesamiento.ipynb
│   └── 03_evaluacion_comentarios_reales.ipynb
├── app/                   # Aplicación Streamlit
│   └── app.py
├── models/                # Modelos entrenados serializados (no versionado)
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

# Lanzar la aplicación
streamlit run app/app.py
```

## Pipeline y resultados

1. **EDA** (`01_exploracion_dataset.ipynb`): sin nulos, 997 filas tras deduplicar,
   variable objetivo `IsToxic` balanceada al 46/54, longitud de comentario muy
   sesgada a la derecha (mediana 19 palabras). Hallazgo relevante: "black" aparece
   en el top 15 de palabras de comentarios tóxicos (113 veces) y no en el de no
   tóxicos, señal de un posible sesgo del dataset a documentar como limitación.
2. **Preprocesamiento** (`02_preprocesamiento.ipynb`): limpieza de texto (minúsculas,
   URLs, caracteres especiales), stopwords personalizadas (se conservan negaciones e
   intensificadores como "not", "never", "very"), tokenización y lematización con
   NLTK.
3. **Vectorización**: TF-IDF (max_features=5000, ngram_range=(1,2), min_df=2).
4. **Modelado**: se compararon Logistic Regression, Naive Bayes, SVM y Random Forest.
   Modelo ganador: **Random Forest** optimizado con GridSearchCV
   (`max_depth=30, min_samples_split=5, n_estimators=100`) → **F1 (clase tóxico) =
   0.67, accuracy = 0.72**.
5. **Evaluación con comentarios reales** (`03_evaluacion_comentarios_reales.ipynb`):
   10 comentarios reales de YouTube (fuente: Sage Journals) con etiqueta esperada
   conocida → **7/10 aciertos**. El modelo detecta el 100% de los no tóxicos y 2/5
   de los tóxicos: acierta cuando hay vocabulario explícitamente ofensivo, pero falla
   con insultos sutiles sin palabrotas ("She is a brat.", "What a spoiled child").
   Esto es coherente con el recall de 0.61 de la clase tóxica y es una limitación
   documentada del modelo (TF-IDF no capta contexto semántico ni tono).

## Roadmap por niveles

### 🟢 Nivel esencial

- [x] Modelo de ML que detecte comentarios tóxicos
- [ ] Overfitting controlado (diferencia train/test < 5 puntos) — pendiente de
      verificación explícita
- [x] Aplicación funcional (Streamlit) para consultar si un mensaje es tóxico
- [x] Repositorio Git organizado, commits limpios y descriptivos
- [x] Documentación y README

### 🟡 Nivel medio

- [ ] Modelo con técnicas de ensemble adicionales (Voting/Stacking) — Random Forest
      ya es un ensemble, se valorará si se amplía
- [ ] Detección a partir de la URL de un vídeo de YouTube
- [ ] Tests unitarios
- [x] Ajuste de hiperparámetros (GridSearchCV)

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
- **Flujo:** Issue → rama de funcionalidad → commit → Pull Request → `developer` →
  `main`.
- **Idiomas:** código, commits y Pull Requests en inglés; documentación en español.
- **Gestión de tareas:** tablero Kanban en GitHub Projects (Backlog → To Do →
  In Progress → Review → Done).

## Fechas clave

- **Entrega:** 22 de septiembre de 2026
- **Presentación (presencial):** 23 de septiembre de 2026

## Estado actual

- [x] Entorno virtual configurado
- [x] Dataset cargado y explorado (estructura, nulos, duplicados, balance de clases)
- [x] Variable objetivo decidida (`IsToxic`)
- [x] EDA completo
- [x] Preprocesamiento NLP
- [x] Vectorización (TF-IDF)
- [x] Entrenamiento y comparación de modelos (4 algoritmos + GridSearchCV)
- [x] Evaluación con comentarios reales de YouTube
- [x] Aplicación Streamlit funcional
- [ ] Verificación explícita de overfitting (train vs test)
- [ ] Nivel medio: tests unitarios, integración por URL de vídeo
