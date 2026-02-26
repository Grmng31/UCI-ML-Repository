"""
Módulo de preprocessing para el proyecto Adult.
Contiene funciones para construir y aplicar transformaciones sklearn
siguiendo las especificaciones exactas de transformación.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer, TargetEncoder
from sklearn.impute import SimpleImputer

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================
BASE_PATH = Path("/content/drive/MyDrive/adult_mlops_project")
PROC_DIR = BASE_PATH / "data" / "processed"
ARTIFACTS_DIR = BASE_PATH / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DEFINICIÓN DE COLUMNAS
# ============================================================================
NUM_COLS = ['age', 'hours-per-week', 'capital-gain', 'capital-loss']
CAT_COLS = ['workclass', 'occupation', 'education', 'marital-status',
            'relationship', 'race', 'sex', 'native-country']
DROP_COLS = ['fnlwgt']  # Será eliminada automáticamente por remainder='drop'

#Codigo de pptx punto 3
def build_preprocessor():
    """
    Construye el ColumnTransformer con las transformaciones específicas
    según la tabla del proyecto:

    - age, hours-per-week: SimpleImputer(media) + StandardScaler
    - capital-gain, capital-loss: SimpleImputer(media) + np.log1p
    - workclass, occupation, [resto cat]: SimpleImputer(moda) + OrdinalEncoder
    - education: SimpleImputer(moda) + TargetEncoder
    - fnlwgt: Eliminada (no incluida en ningún transformer)

    Returns
    -------
    ColumnTransformer
        Preprocesador configurado y listo para entrenar.
    """

    # -------------------------------------------------------------------------
    # 1. Pipeline para variables numéricas con StandardScaler (age, hours-per-week)
    # -------------------------------------------------------------------------
    numeric_scaler_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # -------------------------------------------------------------------------
    # 2. Pipeline para variables con transformación log (capital-gain, capital-loss)
    # -------------------------------------------------------------------------
    numeric_log_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('log_transform', FunctionTransformer(
            np.log1p,
            validate=False,
            feature_names_out='one-to-one'
        ))
    ])

    # -------------------------------------------------------------------------
    # 3. Pipeline para variables categóricas con OrdinalEncoder
    #    (workclass, occupation + resto excepto education)
    # -------------------------------------------------------------------------
    ordinal_cols = [col for col in CAT_COLS if col != 'education']

    ordinal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])

    # -------------------------------------------------------------------------
    # 4. Pipeline para Target Encoding (education)
    # -------------------------------------------------------------------------
    target_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('target_enc', TargetEncoder(target_type='binary'))
    ])

    # -------------------------------------------------------------------------
    # 5. ColumnTransformer final
    # -------------------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_scaler', numeric_scaler_pipe, ['age', 'hours-per-week']),
            ('num_log', numeric_log_pipe, ['capital-gain', 'capital-loss']),
            ('cat_ordinal', ordinal_pipe, ordinal_cols),
            ('cat_target', target_pipe, ['education'])
        ],
        remainder='drop',  # Elimina fnlwgt y cualquier otra columna no especificada
        verbose_feature_names_out=False
    )

    return preprocessor


def preprocess_and_save():
    """
    Ejecuta el pipeline completo de preprocessing:
    1. Carga datos train/test desde data/processed
    2. Ajusta el preprocesador (con y_train para TargetEncoder)
    3. Transforma train y test
    4. Extrae y guarda artefactos individuales:
       - scaler.joblib (StandardScaler)
       - encoder.joblib (OrdinalEncoder)
       - target_enc.joblib (TargetEncoder)
       - pipeline.joblib (ColumnTransformer completo)
    5. Guarda versiones procesadas opcionales

    Raises
    ------
    FileNotFoundError
        Si no encuentra los archivos parquet en PROC_DIR
    Exception
        Cualquier error durante el procesamiento
    """

    try:
        # -------------------------------------------------------------------------
        # Carga de datos
        # -------------------------------------------------------------------------
        print(f"[INFO] Cargando datos desde {PROC_DIR}...")

        X_train = pd.read_parquet(PROC_DIR / "Xtrain.parquet")
        y_train = pd.read_parquet(PROC_DIR / "ytrain.parquet")
        X_test = pd.read_parquet(PROC_DIR / "Xtest.parquet")

        # Asegurar que y_train sea un array 1D (requerido por TargetEncoder)
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0].values
        else:
            y_train = y_train.values.ravel()

        print(f"[INFO] Datos cargados: Train={X_train.shape}, Test={X_test.shape}")

        # -------------------------------------------------------------------------
        # Construcción y entrenamiento del preprocesador
        # -------------------------------------------------------------------------
        print("[INFO] Construyendo y ajustando preprocesador...")
        preprocessor = build_preprocessor()

        # Fit con y_train para permitir TargetEncoder en education
        preprocessor.fit(X_train, y_train)

        # -------------------------------------------------------------------------
        # Transformación de datos
        # -------------------------------------------------------------------------
        print("[INFO] Transformando conjuntos de datos...")
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # -------------------------------------------------------------------------
        # Extracción y guardado de artefactos específicos
        # -------------------------------------------------------------------------
        print(f"[INFO] Guardando artefactos en {ARTIFACTS_DIR}...")

        # 1. StandardScaler (age, hours-per-week)
        scaler = preprocessor.named_transformers_['num_scaler'].named_steps['scaler']
        joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")

        # 2. OrdinalEncoder (workclass, occupation, etc.)
        encoder = preprocessor.named_transformers_['cat_ordinal'].named_steps['ordinal']
        joblib.dump(encoder, ARTIFACTS_DIR / "encoder.joblib")

        # 3. TargetEncoder (education)
        target_enc = preprocessor.named_transformers_['cat_target'].named_steps['target_enc']
        joblib.dump(target_enc, ARTIFACTS_DIR / "target_enc.joblib")

        # 4. Pipeline completo
        joblib.dump(preprocessor, ARTIFACTS_DIR / "pipeline.joblib")

        # -------------------------------------------------------------------------
        # Guardado opcional de datos procesados (para debugging/verificación)
        # -------------------------------------------------------------------------
        # Reconstruir nombres de columnas para el output
        ordinal_cols = [col for col in CAT_COLS if col != 'education']
        output_cols = (['age', 'hours-per-week'] +
                      ['capital-gain', 'capital-loss'] +
                      ordinal_cols +
                      ['education'])

        X_train_proc_df = pd.DataFrame(
            X_train_processed,
            columns=output_cols,
            index=X_train.index
        )
        X_test_proc_df = pd.DataFrame(
            X_test_processed,
            columns=output_cols,
            index=X_test.index
        )

        X_train_proc_df.to_parquet(ARTIFACTS_DIR / "Xtrain_processed.parquet")
        X_test_proc_df.to_parquet(ARTIFACTS_DIR / "Xtest_processed.parquet")

        print("[INFO] Preprocesamiento completado exitosamente.")
        print(f"[INFO] Artefactos guardados:")
        print(f"       - {ARTIFACTS_DIR / 'scaler.joblib'}")
        print(f"       - {ARTIFACTS_DIR / 'encoder.joblib'}")
        print(f"       - {ARTIFACTS_DIR / 'target_enc.joblib'}")
        print(f"       - {ARTIFACTS_DIR / 'pipeline.joblib'}")

    except FileNotFoundError as e:
        print(f"[ERROR] Archivo no encontrado: {e}")
        print(f"[ERROR] Verifique que existan los archivos en {PROC_DIR}")
        raise
    except Exception as e:
        print(f"[ERROR] Error durante el preprocesamiento: {e}")
        raise


if __name__ == "__main__":
    preprocess_and_save()
