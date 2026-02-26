#Esta celda genera `src/evaluate.py` para evaluar los 4 modelos y guardar métricas y reports por separado.

#- Lee `Xtest_processed.parquet` y `ytest.parquet` (asumiendo que `features.py` ya los generó).
#- Calcula métricas (accuracy, F1 macro, ROC-AUC si aplica) para cada modelo.
#- Guarda:
#  - `artifacts/metrics_global.json`
#  - `artifacts/metrics_capital_logit.json`
#  - `artifacts/metrics_time_linear.json`
#  - `artifacts/metrics_work_binary.json`
#  - Un `artifacts/metrics_all_models.json` agregando todo.

import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Definición de rutas base del proyecto
BASE_PATH = Path("/content/drive/MyDrive/adult_mlops_project") # Corrected path to use underscores
PROC_DIR = BASE_PATH / "data" / "processed"
MODELS_DIR = BASE_PATH / "models"
ARTIFACTS_DIR = BASE_PATH / "artifacts"


def evaluate_all_models() -> dict:
    """
    Evalúa los cuatro modelos entrenados del proyecto Adult Income usando el conjunto de test procesado y guarda las métricas en JSON.

    Returns:
        dict con las métricas agregadas por modelo.
    """
    # Asegurar que el directorio de artifacts exista
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Carga de datos de test
    try:
        X_test = pd.read_parquet(ARTIFACTS_DIR / "Xtest_processed.parquet")
        y_test = pd.read_parquet(PROC_DIR / "ytest.parquet")
    except Exception as e:
        raise FileNotFoundError(f"Error al cargar los datos de test: {e}. Verifica que existan los archivos en {ARTIFACTS_DIR} y {PROC_DIR}")

    # Convertir y_test a pd.Series si es DataFrame
    if isinstance(y_test, pd.DataFrame):
        if y_test.shape[1] == 1:
            y_test = y_test.squeeze()
        else:
            raise ValueError("y_test tiene múltiples columnas, se esperaba una única columna de etiquetas.")

    # Definición de los modelos y sus subconjuntos de features correspondientes
    model_configs = [
        {
            "name": "global",
            "file": "model_global.pkl",
            "features": X_test.columns.tolist(),  # Todas las columnas
            "json_out": "metrics_global.json"
        },
        {
            "name": "capital_logit",
            "file": "model_capital_logit.pkl",
            "features": ["capital-gain", "capital-loss"],
            "json_out": "metrics_capital_logit.json"
        },
        {
            "name": "time_linear",
            "file": "model_time_linear.pkl",
            "features": ["age", "hours-per-week"],
            "json_out": "metrics_time_linear.json"
        },
        {
            "name": "work_binary",
            "file": "model_work_binary.pkl",
            "features": ["workclass", "occupation"],
            "json_out": "metrics_work_binary.json"
        }
    ]

    all_results = {}

    for config in model_configs:
        model_name = config["name"]
        model_path = MODELS_DIR / config["file"]

        print(f"\nEvaluando modelo: {model_name}...")

        # Validar existencia del archivo de modelo
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

        # Cargar modelo
        try:
            model = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo {config['file']}: {e}")

        # Validar y seleccionar subconjunto de features
        required_features = config["features"]
        missing_cols = set(required_features) - set(X_test.columns)
        if missing_cols:
            raise ValueError(f"Faltan las siguientes columnas en X_test para el modelo {model_name}: {missing_cols}")

        X_subset = X_test[required_features]

        # Realizar predicciones
        try:
            y_pred = model.predict(X_subset)
        except Exception as e:
            raise RuntimeError(f"Error al predecir con el modelo {model_name}: {e}")

        # Calcular métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")

        # Generar classification report como diccionario
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Calcular ROC AUC si es posible (manejo de excepciones)
        roc_auc = None
        try:
            if hasattr(model, "predict_proba"):
                # Para clasificación binaria, tomamos la probabilidad de la clase positiva (columna 1)
                y_proba = model.predict_proba(X_subset)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_subset)
                roc_auc = roc_auc_score(y_test, y_score)
        except Exception:
            # Si no se puede calcular (ej. multiclase sin especificar, o error en forma de datos), se deja como None
            roc_auc = None

        # Preparar diccionario de métricas
        metrics = {
            "model_name": model_name,
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "classification_report": class_report
        }

        # Guardar métricas individuales en JSON
        individual_json_path = ARTIFACTS_DIR / config["json_out"]
        try:
            with open(individual_json_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"Métricas guardadas en: {individual_json_path}")
        except Exception as e:
            raise IOError(f"Error al guardar métricas individuales para {model_name}: {e}")

        # Agregar al resultado agregado
        all_results[model_name] = metrics

    # Guardar JSON agregado con todas las métricas
    aggregated_json_path = ARTIFACTS_DIR / "metrics_all_models.json"
    try:
        with open(aggregated_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nMétricas agregadas guardadas en: {aggregated_json_path}")
    except Exception as e:
        raise IOError(f"Error al guardar el archivo de métricas agregadas: {e}")

    return all_results


if __name__ == "__main__":
    try:
        results = evaluate_all_models()
        print("\n" + "="*50)
        print("Evaluación completada. Resumen de resultados:")
        print("="*50)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"\nFallo en la ejecución de evaluate.py: {e}")
        raise
