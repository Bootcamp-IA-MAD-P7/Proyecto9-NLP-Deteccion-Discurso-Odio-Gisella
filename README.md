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
  dataset razonablemente balanceado). Se descartó `IsHatespeech`, al ser una
  categoría más específica y menos representada dentro del dataset; `IsToxic`
  ofrece una señal más completa para el problema de negocio.
- Se valoró ampliar el dataset (otro dataset público, web scraping o data
  augmentation), pero el EDA no mostró una limitación real que lo justificara, por lo
  que se mantiene el dataset original.

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
| Vectorización                       | TF-IDF (unigramas)                                                   |
| Machine Learning                    | scikit-learn (Regresión Logística, Naive Bayes, SVM, Random Forest) |
| Ajuste de hiperparámetros           | GridSearchCV                                                        |
| Ensemble                            | VotingClassifier (hard voting: Regresión Logística + Naive Bayes + SVM) — modelo final |
| Testing                             | pytest                                                              |
| Integración YouTube                 | youtube-comment-downloader                                          |
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
│   ├── app.py
│   ├── utils.py           # Limpieza de texto y predicción (importable, sin efectos de Streamlit)
│   └── youtube_utils.py   # Descarga y análisis en lote de comentarios de YouTube
├── tests/                 # Tests unitarios (pytest)
│   ├── conftest.py
│   └── test_utils.py
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
```

> **Nota:** el dataset no se incluye en este repositorio. Para reproducir el
> análisis, descarga `youtoxic_english_1000.csv` y colócalo dentro de la carpeta
> `data/`.

```bash
# Lanzar la aplicación
streamlit run app/app.py

# Ejecutar los tests unitarios
pytest tests/ -v
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
3. **Vectorización y detección de overfitting**: la configuración inicial de TF-IDF
   (max_features=5000, ngram_range=(1,2), min_df=2) generaba una alta
   dimensionalidad (2646 variables para solo 797 comentarios de entrenamiento), lo
   que producía overfitting severo (16-37 puntos de diferencia train/test) en los
   cuatro modelos comparados. Se ajustó a **max_features=1000, ngram_range=(1,1)
   (solo unigramas), min_df=5**, reduciendo el problema aunque sin eliminarlo del
   todo.
4. **Metodología de evaluación**: en lugar de comparar el rendimiento en train "en
   bruto" contra test, se comparó el **F1 de validación cruzada (5-fold) frente al
   F1 de test**, una medida más rigurosa y justa de la capacidad de generalización
   (usa en cada fold datos no vistos durante el entrenamiento).
5. **Modelado y selección del modelo final**: se compararon Logistic Regression,
   Naive Bayes, SVM y Random Forest (este último optimizado con GridSearchCV). El
   modelo final es un **ensemble VotingClassifier (hard voting)** de Logistic
   Regression, Naive Bayes y SVM. Random Forest quedó descartado por peor
   overfitting (8.60 puntos CV vs test) y peor F1 en test (0.57). Resultado del
   ensemble: **F1 (test) = 0.6322**, overfitting CV vs test = **4.64 puntos**
   (dentro del umbral de 5 puntos definido como aceptable).
6. **Evaluación con comentarios reales** (`03_evaluacion_comentarios_reales.ipynb`):
   10 comentarios reales de YouTube (fuente: Sage Journals) con etiqueta esperada
   conocida. Se detectó y corrigió una inconsistencia en el pipeline de predicción
   (`predecir_toxicidad` no aplicaba `clean_text()` antes de tokenizar, a diferencia
   del pipeline de entrenamiento); tras corregirla, el resultado real es **2 de 5
   comentarios tóxicos detectados**. El modelo acierta cuando hay vocabulario
   explícitamente ofensivo, pero falla con insultos sutiles o coloquiales sin
   palabrotas explícitas. Es una limitación documentada y coherente con el F1 de
   0.63 de la clase tóxica (TF-IDF no capta contexto semántico ni tono).
7. **Tests unitarios** (`tests/test_utils.py`): 14 tests con pytest sobre
   `clean_text`, `tokenize_and_lemmatize` y `predecir_toxicidad`, usando dobles de
   prueba para el ensemble y el vectorizador, de forma que los tests no dependen de
   los modelos ya entrenados.
8. **Integración con YouTube** (`app/youtube_utils.py`): dado el enlace de un
   vídeo, se extrae el ID, se descargan sus comentarios reales con
   `youtube-comment-downloader` y se analizan en lote con el mismo pipeline de
   predicción. Integrado en la app de Streamlit con un campo de URL, un selector
   del número de comentarios a analizar y una tabla de resultados.

## Roadmap por niveles

### 🟢 Nivel esencial

- [x] Modelo de ML que detecte comentarios tóxicos
- [x] Overfitting controlado y documentado (comparación F1 CV vs test, 4.64 puntos
      de diferencia en el modelo final, dentro del umbral de 5 puntos)
- [x] Aplicación funcional (Streamlit) para consultar si un mensaje es tóxico
- [x] Repositorio Git organizado, commits limpios y descriptivos
- [x] Documentación y README

### 🟡 Nivel medio

- [x] Modelo con técnicas de ensemble (VotingClassifier: Logistic Regression +
      Naive Bayes + SVM)
- [x] Detección de comentarios tóxicos a partir de la URL de un vídeo de YouTube
- [x] Tests unitarios (pytest)
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
- **Gestión de tareas:** tablero Kanban en GitHub Projects (Todo → In Progress →
  Done).

## Estado actual

- [x] Entorno virtual configurado
- [x] Dataset cargado y explorado (estructura, nulos, duplicados, balance de clases)
- [x] Variable objetivo decidida (`IsToxic`)
- [x] EDA completo
- [x] Preprocesamiento NLP
- [x] Vectorización (TF-IDF) ajustada tras detectar overfitting
- [x] Entrenamiento y comparación de modelos (4 algoritmos + GridSearchCV)
- [x] Modelo final: ensemble VotingClassifier (hard voting)
- [x] Overfitting verificado y documentado (F1 CV vs test)
- [x] Evaluación con comentarios reales de YouTube (pipeline corregido)
- [x] Aplicación Streamlit funcional (análisis de comentario individual)
- [x] Tests unitarios (pytest) sobre `utils.py`
- [x] Integración con YouTube: análisis en lote de comentarios reales por URL
