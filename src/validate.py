import pandas as pd
import numpy as np
import pandera as pa
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List
import traceback # Import traceback module
from ucimlrepo import fetch_ucirepo # Add this import statement

# – Generar src/validate.py
# --- CÓDIGO DEL PPTX (Punto 1) ---
def ingest_adult(output_dir: str = 'drive/MyDrive/adult_mlops_project/data/raw') -> dict:
    adult = fetch_ucirepo(id=2)
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Guardamos en formato parquet como menciona el PPTX
    adult.data.features.to_parquet(path / 'features.parquet')
    adult.data.targets.to_parquet(path / 'targets.parquet')

    return {
        'n_rows': len(adult.data.features),
        'n_features': adult.data.features.shape[1],
        'target_dist': adult.data.targets.value_counts().to_dict()
    }

# Ejecución de la ingesta
stats = ingest_adult()
print(f"Datos ingeridos: {stats}")

# Constantes de rutas para Google Drive
BASE_PATH = Path("/content/drive/MyDrive/adult_mlops_project")
RAW_DIR = BASE_PATH / "data" / "raw"
ARTIFACTS_DIR = BASE_PATH / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Esquema de validación con Pandera para el dataset Adult (14 columnas)
SCHEMA = pa.DataFrameSchema({
    "age": pa.Column(int, checks=pa.Check.in_range(17, 90)),
    "workclass": pa.Column(str, nullable=True),
    "fnlwgt": pa.Column(int, checks=pa.Check.greater_than(0)),
    "education": pa.Column(str),
    "education-num": pa.Column(int, checks=pa.Check.in_range(1, 16)),
    "marital-status": pa.Column(str),
    "occupation": pa.Column(str, nullable=True),
    "relationship": pa.Column(str),
    "race": pa.Column(str),
    "sex": pa.Column(str),
    "capital-gain": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
    "capital-loss": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
    "hours-per-week": pa.Column(int, checks=pa.Check.in_range(1, 99)),
    "native-country": pa.Column(str, nullable=True)
})

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> List[int]:
    """
    Detecta outliers usando el método IQR (rango intercuartílico).

    Args:
        df: DataFrame con los datos numéricos
        column: Nombre de la columna a analizar

    Returns:
        Lista de índices donde se detectaron outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_idx = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
    return outliers_idx

def validate_data(features_path: Path, targets_path: Path, output_report_path: Path) -> dict:
    """
    Valida los datos raw usando Pandera y genera un JSON de reporte.

    Args:
        features_path: Ruta al archivo parquet de características (X)
        targets_path: Ruta al archivo parquet de objetivos (y)
        output_report_path: Ruta donde se guardará el reporte JSON

    Returns:
        Diccionario con el reporte completo de validación
    """
    print(f"Iniciando proceso de validación de datos...")
    print(f"Cargando features desde: {features_path}")
    print(f"Cargando targets desde: {targets_path}")

    report = {
        "schema_valid": False,
        "nulls_pct": {},
        "outliers": {},
        "duplicates": 0,
        "timestamp": datetime.now().isoformat(),
        "n_rows": 0,
        "n_features": 0
    }
    try:
        # 1. Cargar data/raw/*.parquet
        X = pd.read_parquet(features_path)
        y = pd.read_parquet(targets_path)

        # Verificar alineación entre features y targets
        if len(X) != len(y):
            raise ValueError(f"Desalineación de datos: {len(X)} features vs {len(y)} targets")
        
        # Removed redundant call to ingest_adult()
        # stats = ingest_adult() 
        # print(f"Datos ingeridos: {stats}")

        # Combinar para análisis de duplicados y nulos
        df_combined = X.copy()
        target_col_name = y.columns[0] if len(y.columns) > 0 else "income"
        df_combined[target_col_name] = y.values.ravel() # Fix: Use .ravel() to flatten the 2D array

        report["n_rows"] = int(len(X))
        report["n_features"] = int(X.shape[1])

        print(f"Datos cargados exitosamente: {report['n_rows']} filas, {report['n_features']} características")

        # Limpieza básica de etiquetas en el target (a veces traen un punto final '.')
        # Modificación: Extraer el string de la tupla si es necesario
        if y.dtypes.iloc[0] == 'object' and y.iloc[0,0] and isinstance(y.iloc[0,0], tuple):
            y[target_col_name] = y[target_col_name].apply(lambda x: x[0] if isinstance(x, tuple) else x)
        y[target_col_name] = y[target_col_name].str.replace('.', '', regex=False)

        # 2. Validar con Pandera
        print("Ejecutando validación de schema...")
        try:
            SCHEMA.validate(X, lazy=True)
            report["schema_valid"] = True
            print("✓ Validación de schema completada sin errores")
        except pa.errors.SchemaErrors as e:
            report["schema_valid"] = False
            print(f"✗ Errores de schema detectados en {len(e.failure_cases)} registros")
            print(f"  Detalles: {e.failure_cases[['column', 'check', 'failure_case']].head()}")

        # 3. Calcular % nulos por columna
        print("Calculando porcentaje de valores nulos...")
        nulls_count = df_combined.isnull().sum()
        nulls_percentage = (nulls_count / len(df_combined) * 100).round(4)

        # Solo incluir columnas con nulos > 0
        nulls_dict = nulls_percentage[nulls_count > 0].to_dict()
        report["nulls_pct"] = nulls_dict if nulls_dict else {}

        if nulls_dict:
            print(f"  Nulos detectados en {len(nulls_dict)} columnas")
        else:
            print("  No se detectaron valores nulos")

        # 4. Detectar outliers (IQR) por columna numérica
        print("Detectando outliers usando método IQR...")
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        outliers_dict = {}

        for col in numeric_columns:
            outlier_indices = detect_outliers_iqr(X, col)
            if outlier_indices:
                # Limitar a primeros 50 índices para mantener JSON manejable
                outliers_dict[col] = outlier_indices[:50]
                print(f"  - {col}: {len(outlier_indices)} outliers detectados")

        report["outliers"] = outliers_dict

        # 5. Contar duplicados
        print("Verificando registros duplicados...")
        duplicates_count = df_combined.duplicated().sum()
        report["duplicates"] = int(duplicates_count)
        print(f"  Duplicados encontrados: {duplicates_count}")

        # 6. Crear artifacts/validation_report.json
        print(f"Generando reporte JSON en: {output_report_path}")
        output_report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print("✓ Proceso de validación completado exitosamente")

    except FileNotFoundError as e:
        error_msg = f"Archivo no encontrado: {str(e)}"
        print(f"✗ Error: {error_msg}")
        report["error"] = error_msg
    except Exception as e:
        error_msg = f"Error durante la validación: {str(e)}"
        print(f"✗ Error inesperado: {error_msg}")
        print(traceback.format_exc()) # Print full traceback
        report["error"] = error_msg

        # Intentar guardar reporte de error
        try:
            output_report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    return report

if __name__ == "__main__":
    """Ejecución como script para validación de datos Adult."""
    print("=" * 60)
    print("MÓDULO DE VALIDACIÓN DE DATOS - ADULT MLOPS PROJECT")
    print("=" * 60)

    # Configuración de rutas para ejecución directa
    features_path = RAW_DIR / "features.parquet"
    targets_path = RAW_DIR / "targets.parquet"
    report_path = ARTIFACTS_DIR / "validation_report.json"

    # Verificar existencia de archivos
    if not features_path.exists():
        print(f"ERROR: No se encuentra el archivo: {features_path}")
        print("Asegúrate de ejecutar la ingesta de datos primero.")
    elif not targets_path.exists():
        print(f"ERROR: No se encuentra el archivo: {targets_path}")
    else:
        # Ejecutar validación
        resultado = validate_data(features_path, targets_path, report_path)

        # Resumen final
        print("\n" + "=" * 60)
        print("RESUMEN DE VALIDACIÓN")
        print("=" * 60)
        print(f"Timestamp:       {resultado['timestamp']}")
        print(f"Filas validadas: {resultado['n_rows']}")
        print(f"Features:        {resultado['n_features']}")
        print(f"Schema válido:   {'SÍ' if resultado['schema_valid'] else 'NO'}")
        print(f"Duplicados:      {resultado['duplicates']}")
        print(f"Columnas con nulos: {len(resultado['nulls_pct'])}")
        print(f"Columnas con outliers: {len(resultado['outliers'])}")
        print(f"Reporte guardado: {report_path}")
        print("=" * 60)
