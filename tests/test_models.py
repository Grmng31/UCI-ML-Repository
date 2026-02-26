#– `tests/test_models.py` (almacenar outputs de evaluación)

#Hacemos un pequeño “test runner” que:

#- Lee `artifacts/metrics_all_models.json`.
#- Imprime un resumen.
#- Opcional: falla si algún modelo está por debajo de un umbral (por ejemplo F1 < 0.75).
#- Guarda una copia en `tests/last_test_run.json`.
import json
from pathlib import Path

# Definición de rutas base del proyecto MLOps
BASE_PATH = Path("/content/drive/MyDrive/adult_mlops_project")
ARTIFACTS_DIR = BASE_PATH / "artifacts"
TESTS_DIR = BASE_PATH / "tests"

# Asegurar la existencia del directorio de tests
TESTS_DIR.mkdir(parents=True, exist_ok=True)


def check_models(min_f1: float = 0.75) -> dict:
    """
    Lee artifacts/metrics_all_models.json y valida que cada modelo supere
    un umbral mínimo de F1-macro.

    Args:
        min_f1: Umbral mínimo aceptable de F1-macro por modelo.

    Returns:
        dict con:
        - "passed": True/False (indica si todos los modelos cumplen el umbral)
        - "details": métricas por modelo con estado individual
    """
    metrics_path = ARTIFACTS_DIR / "metrics_all_models.json"

    # Verificar existencia del artefacto de métricas
    if not metrics_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de métricas: {metrics_path}")

    # Carga de métricas generadas por src/evaluate.py
    with open(metrics_path, "r", encoding="utf-8") as f:
        all_metrics = json.load(f)

    details = {}
    all_passed = True

    # Iteración sobre cada modelo evaluado
    for model_name, metrics in all_metrics.items():
        model_result = {
            "metrics": metrics,
            "f1_macro": None,
            "passed": False,
            "warning": None
        }

        # Extracción de F1-macro con manejo de ausencia
        try:
            f1_score = metrics.get("f1_macro")
            if f1_score is None:
                raise KeyError("f1_macro no encontrado")

            model_result["f1_macro"] = float(f1_score)

            # Validación contra umbral mínimo
            if model_result["f1_macro"] >= min_f1:
                model_result["passed"] = True
            else:
                model_result["passed"] = False
                all_passed = False

        except (KeyError, TypeError, ValueError) as e:
            # Marcar warning si falta la métrica o tiene formato inválido
            model_result["warning"] = f"Metrica f1_macro no disponible o inválida: {str(e)}"
            model_result["passed"] = False
            all_passed = False

        details[model_name] = model_result

    # Construcción del resultado global
    result = {
        "passed": all_passed,
        "threshold": min_f1,
        "total_models": len(details),
        "models_passed": sum(1 for d in details.values() if d["passed"]),
        "details": details
    }

    # Persistencia del reporte de test en el directorio de tests
    output_path = TESTS_DIR / "last_test_run.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    # Ejecución del test runner como script standalone
    result = check_models()

    print("Resultado de tests sobre modelos:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Validación final: exit con error si no se cumplen los criterios
    # if not result.get("passed", False):
    #     raise SystemExit("❌ Algunos modelos no alcanzan el F1 mínimo requerido.")
    # else:
    #     print("✅ Todos los modelos cumplen el umbral de F1.")
