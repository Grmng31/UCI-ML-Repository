#Generar src/ingest.py
import pandas as pd
from pathlib import Path
from ucimlrepo import fetch_ucirepo
from typing import Optional, Dict, Any

# Configuración de rutas para Google Drive
BASE_PATH = Path("/content/drive/MyDrive/adult_mlops_project")
RAW_DIR = BASE_PATH / "data" / "raw"


def ingest_adult(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Descarga el dataset Adult desde UCI ML Repository y lo guarda en Parquet.

    Args:
        output_dir: Directorio donde guardar los datos. Si es None, usa BASE_PATH/data/raw.

    Returns:
        dict con metadatos:
            - n_rows: número de filas
            - n_features: número de features
            - target_dist: distribución de la variable objetivo
    """
    try:
        # Determinar directorio de salida
        if output_dir is None:
            output_dir = RAW_DIR

        # Crear directorio si no existe
        output_dir.mkdir(parents=True, exist_ok=True)

        # Obtener dataset Adult (ID=2)
        print("Descargando dataset Adult desde UCI ML Repository...")
        adult = fetch_ucirepo(id=2)

        # Extraer features y targets
        X = adult.data.features
        y = adult.data.targets

        # Guardar en formato Parquet
        features_path = output_dir / "features.parquet"
        targets_path = output_dir / "targets.parquet"

        X.to_parquet(features_path)
        y.to_parquet(targets_path)

        print(f"Features guardadas en: {features_path}")
        print(f"Targets guardados en: {targets_path}")

        # Calcular metadatos
        stats = {
            'n_rows': len(adult.data.features),
            'n_features': adult.data.features.shape[1],
            'target_dist': adult.data.targets.value_counts().to_dict()
        }

        # Imprimir resumen del dataset
        print("\n=== Metadatos del Dataset ===")
        print(adult.metadata)
        print("\n=== Información de Variables ===")
        print(adult.variables)

        return stats

    except Exception as e:
        print(f"Error durante la ingesta de datos: {e}")
        raise


if __name__ == "__main__":
    try:
        stats = ingest_adult()
        print("\nIngesta completada. Estadísticas:")
        print(stats)
    except Exception as e:
        print(f"Fallo en la ejecución del script: {e}")
