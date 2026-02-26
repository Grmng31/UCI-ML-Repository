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

        #– Generar `src/clean.py` (clean → `data/processed` y `data/interim`)

#Esta celda genera un módulo `src/clean.py` que replique la lógica de limpieza de tu notebook (reemplazo `'?'`, `drop_duplicates`, sincronizar `y`, limpieza de `income`, split 80/20, guardado en `processed` + copia a `interim`).[3][2]

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
from typing import Dict, Any
import sys

# Definición de rutas base del proyecto
BASE_PATH = Path("/content/drive/MyDrive/adult_mlops_project")
RAW_DIR = BASE_PATH / "data" / "raw"
PROC_DIR = BASE_PATH / "data" / "processed"
INT_DIR = BASE_PATH / "data" / "interim"

def clean_and_split() -> Dict[str, Any]:
    """
    Limpia el dataset Adult y genera splits train/test en processed + interim.

    Realiza las siguientes operaciones:
    1. Carga features y targets desde data/raw
    2. Reemplaza '?' por NaN y elimina duplicados manteniendo sincronía entre X e y
    3. Limpia etiquetas de income (elimina puntos finales y espacios en blanco)
    4. Ejecuta partición estratificada 80/20 para train/test
    5. Persiste archivos parquet en processed y genera copia de respaldo en interim

    Returns:
        dict: Diccionario con métricas del proceso (conteos y distribuciones de clases)
    """

    # Crear directorios de salida si no existen
    for d in (PROC_DIR, INT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    try:
        # B. Lectura de datos raw
        print(f"📂 Cargando datos desde {RAW_DIR}...")
        features_path = RAW_DIR / "features.parquet"
        targets_path = RAW_DIR / "targets.parquet"

        if not features_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de features: {features_path}")
        if not targets_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de targets: {targets_path}")

        X = pd.read_parquet(features_path)
        y = pd.read_parquet(targets_path)

        n_rows_raw = len(X)
        print(f"✅ Datos cargados exitosamente: {n_rows_raw} filas")

        # C. Lógica de limpieza

        # Reemplazar "?" por NaN según lógica del notebook
        print("🧹 Reemplazando valores '?' por NaN...")
        X = X.replace("?", np.nan)

        # Eliminar duplicados en X y mantener sincronía con y
        print("🧹 Eliminando filas duplicadas...")
        filas_antes = len(X)
        X = X.drop_duplicates()
        y = y.loc[X.index]  # Mantener alineación de índices
        n_rows_clean = len(X)
        print(f"✅ Limpieza completada: {filas_antes - n_rows_clean} duplicados eliminados")

        # Limpiar la etiqueta income (quitar punto final y espacios)
        if "income" in y.columns:
            print("🧹 Normalizando etiquetas de income...")
            y["income"] = y["income"].str.replace(".", "", regex=False).str.strip()
        else:
            print("⚠️ Advertencia: No se encontró columna 'income' en el dataset de targets")

        # D. Partición train/test estratificada 80/20
        print("✂️ Realizando partición estratificada (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        n_train = len(X_train)
        n_test = len(X_test)
        print(f"✅ Split completado: {n_train} train / {n_test} test")

        # Calcular distribuciones de clases
        if "income" in y_train.columns:
            class_dist_train = y_train["income"].value_counts().to_dict()
            class_dist_test = y_test["income"].value_counts().to_dict()
        else:
            class_dist_train = {}
            class_dist_test = {}

        # E. Guardado en processed (formato Parquet)
        print(f"💾 Guardando datasets procesados en {PROC_DIR}...")

        X_train.to_parquet(PROC_DIR / "Xtrain.parquet")
        X_test.to_parquet(PROC_DIR / "Xtest.parquet")
        y_train.to_parquet(PROC_DIR / "ytrain.parquet")
        y_test.to_parquet(PROC_DIR / "ytest.parquet")

        print("✅ Archivos guardados en data/processed/")

        # F. Copia de respaldo a interim
        print(f"📦 Generando backup en {INT_DIR}...")
        for archivo_parquet in PROC_DIR.glob("*.parquet"):
            shutil.copy2(archivo_parquet, INT_DIR / archivo_parquet.name)
        print("✅ Backup completado en data/interim/")

        # Preparar métricas de retorno
        resultado = {
            "n_rows_raw": n_rows_raw,
            "n_rows_clean": n_rows_clean,
            "n_train": n_train,
            "n_test": n_test,
            "class_dist_train": class_dist_train,
            "class_dist_test": class_dist_test
        }

        # Reporte final
        print("\n" + "="*50)
        print("📊 RESUMEN DEL PROCESO DE LIMPIEZA")
        print("="*50)
        print(f"Filas originales:              {n_rows_raw}")
        print(f"Filas después de limpieza:     {n_rows_clean}")
        print(f"Registros entrenamiento:       {n_train} ({n_train/n_rows_clean*100:.1f}%)")
        print(f"Registros evaluación:          {n_test} ({n_test/n_rows_clean*100:.1f}%)")
        print(f"Distribución clases (train):   {class_dist_train}")
        print(f"Distribución clases (test):    {class_dist_test}")
        print("="*50)

        return resultado

    except Exception as e:
        print(f"❌ Error crítico en el pipeline de limpieza: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        metricas = clean_and_split()
        print("\n✅ Pipeline de limpieza ejecutado exitosamente")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fallo en la ejecución del módulo: {e}")
        sys.exit(1)
#The SystemExit: 0 you're seeing is not an error! It's actually a signal that the script completed its execution successfully.
#The clean_and_split() function in the cell ran all its operations as expected, performing the data cleaning and splitting,
#and then gracefully exited
#with a success status, as confirmed by the message '✅ Pipeline de limpieza ejecutado exitosamente' in the output.

#clean.py
# para ejecutar el módulo y llenar `processed` + `interim`:
##Ejecutar limpieza y splits

import sys
sys.path.append(str(BASE_PATH))

# --- FIX: Ensure src/clean.py has correct `if __name__ == "__main__":` block ---
clean_file_path = BASE_PATH / "src" / "clean.py"
if clean_file_path.exists():
    with open(clean_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the line with `if __name__ == "__main__":`
    main_block_start_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "if __name__ == \"__main__\":":
            main_block_start_idx = i
            break

    if main_block_start_idx != -1:
        # Check if the next line is indented or if it's the end of the file
        if main_block_start_idx + 1 == len(lines) or not lines[main_block_start_idx + 1].startswith(' ' * 4):
            # Insert the missing execution block
            print('🔧 Fixing src/clean.py: Adding missing code under `if __name__ == "__main__":`')
            fix_code = [
                "    try:\n",
                "        stats = clean_and_split()\n",
                "        print('\\nLimpieza y splits completados. Estadísticas:')\n", # Escaped backslash for literal \n in file
                "        print(stats)\n",
                "    except Exception as e:\n",
                "        print(f\"Fallo en la ejecución del script de limpieza: {e}\")\n"
            ]
            # Replace the placeholder (if any) or simply append if block was empty
            # For now, let's just append right after the if statement, ensuring it's indented.
            lines.insert(main_block_start_idx + 1, ''.join(fix_code))

            with open(clean_file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        else:
            print('src/clean.py already seems to have content under `if __name__ == "__main__":`')
    else:
        print('Could not find `if __name__ == "__main__":` block in src/clean.py')
else:
    print("Error: src/clean.py not found!")

# --- End of FIX ---

from src.clean import clean_and_split

stats_clean = clean_and_split()
print("\n✅ Limpieza y splits completados:")
print(stats_clean)

print("\n📁 data/processed:")
for p in (BASE_PATH / "data" / "processed").glob("*.parquet"):
    print(" -", p.name)

print("\n📁 data/interim:")
for p in (BASE_PATH / "data" / "interim").glob("*.parquet"):
    print(" -", p.name)
