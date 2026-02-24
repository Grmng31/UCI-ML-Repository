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
