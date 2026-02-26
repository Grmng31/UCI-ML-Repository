# Adult Income MLOps - Pipeline Multimodelo

Proyecto de Machine Learning Operations (MLOps) para la clasificacion de ingresos (Adult Income Dataset) utilizando una arquitectura multimodelo. El pipeline incluye validacion de datos, ingenieria de caracteristicas, entrenamiento de multiples modelos (uno global y tres especializados), evaluacion comparativa y tests automatizados.

## Descripcion

Este proyecto implementa un flujo completo de MLOps que va desde la ingestion de datos del repositorio UCI hasta la evaluacion de modelos en produccion. La arquitectura multimodelo permite capturar patrones especificos en subconjuntos de datos mediante modelos especializados, complementando al modelo global.

**Caracteristicas principales:**
- Validacion de esquemas con Pandera
- Seguimiento de experimentos con MLflow
- Orquestacion de pipelines con DVC (Data Version Control)
- Contenerizacion con Docker para reproducibilidad total
- Tests automatizados de calidad de modelos

## Estructura de Carpetas

```
adult-mlops-project/
data/
    raw/                    # Datos crudos descargados (features.parquet, targets.parquet)
    processed/              # Datos procesados y listos para entrenamiento
src/
    __init__.py
    ingest.py               # Descarga y almacenamiento inicial
    validate.py             # Validacion de esquemas y calidad de datos
    features.py             # Preprocesamiento y feature engineering
    train.py                # Entrenamiento del modelo global + 3 especializados
    evaluate.py             # Evaluacion comparativa de los 4 modelos
models/                     # Artefactos de modelos entrenados (.pkl)
artifacts/                  # Reportes de validacion, metras y preprocesadores
tests/
    test_models.py          # Tests de validacion de performance y calidad
pyproject.toml              # Dependencias y configuracion del proyecto (Poetry)
dvc.yaml                    # Definicion del pipeline de DVC
Dockerfile                  # Configuracion de contenedor
README.md                   # Este archivo
```

## Como Correr el Pipeline

### 1. Local con Python

Instalar dependencias:
```bash
pip install .
```

Ejecutar etapas individuales:
```bash
python -m src.ingest
python -m src.validate
python -m src.features
python -m src.train
python -m src.evaluate
python -m tests.test_models
```

### 2. Con DVC (Recomendado)

Ejecutar el pipeline completo con seguimiento de dependencias:
```bash
dvc repro
```

Ver el DAG del pipeline:
```bash
dvc dag
```

### 3. Con Docker

Construir la imagen:
```bash
docker build -t adult-mlops-multimodel .
```

Ejecutar entrenamiento:
```bash
docker run --rm -v $(pwd)/models:/app/models adult-mlops-multimodel
```

Ejecutar evaluacion:
```bash
docker run --rm adult-mlops-multimodel python -m src.evaluate
```

Ejecutar tests:
```bash
docker run --rm adult-mlops-multimodel python -m tests.test_models
```

## Arquitectura de Modelos

El sistema entrena y evalua **4 modelos** que operan sobre diferentes subconjuntos de caracteristicas:

### 1. Modelo Global (`model_global.pkl`)
- **Tipo:** GradientBoostingClassifier
- **Descripcion:** Modelo principal entrenado sobre todas las caracteristicas procesadas (numericas y categoricas). Sirve como baseline y predictor general.
- **Uso:** Prediccion estandar cuando no hay informacion especifica sobre el subconjunto de datos.

### 2. Modelo Capital (`model_capital_logit.pkl`)
- **Tipo:** LogisticRegression
- **Descripcion:** Especializado en las variables `capital-gain` y `capital-loss`. Captura relaciones lineales especificas del patrimonio financiero.
- **Uso:** Casos donde las ganancias/perdidas de capital son los predictores dominantes.

### 3. Modelo Tiempo (`model_time_linear.pkl`)
- **Tipo:** LinearRegression (o Ridge/Lasso segun implementacion)
- **Descripcion:** Entrenado sobre `age` y `hours-per-week`. Modela la relacion entre edad, horas trabajadas e ingresos.
- **Uso:** Analisis demografico y laboral puro.

### 4. Modelo Trabajo (`model_work_binary.pkl`)
- **Tipo:** Clasificador Binario (ej. LogisticRegression o RandomForest)
- **Descripcion:** Especializado en variables categoricas de empleo: `workclass` y `occupation`.n- **Uso:** Clasificacion basada unicamente en el sector laboral y tipo de ocupacion.

## Metricas y Tests

### Evaluacion (`src/evaluate.py`)
El script de evaluacion carga los 4 modelos entrenados y los prueba contra el conjunto de test (`Xtest_processed.parquet`). Genera:
- Metricas individuales por modelo (Accuracy, Precision, Recall, F1-Score)
- Comparativa consolidada en `artifacts/metrics_all_models.json`
- Analisis de drift entre entrenamiento y test (si aplica)

### Tests (`tests/test_models.py`)
Suite de validacion que verifica:
- **Thresholds de performance:** F1-Score minimo por modelo (ej. > 0.75)
- **Estabilidad:** Varianza de predicciones dentro de rangos aceptables
- **Integridad:** Los modelos cargan correctamente y generan predicciones del shape esperado

Resultados almacenados en `tests/last_test_run.json`.

```bash
# Correr tests
python -m tests.test_models