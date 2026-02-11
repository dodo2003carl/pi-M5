import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler
import os

def run_feature_engineering(path='../Base_de_datos.xlsx', target='Pago_atiempo'):
    """
    Ingeniería de características (v1.1.0). 
    Ruta ajustada para buscar el Excel fuera de la carpeta 'src'.
    """
    # 1. Verificar si el archivo existe para dar un mensaje claro
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No se encontró el archivo en: {os.path.abspath(path)}")

    # 2. Carga de datos
    df = pd.read_excel(path)
    
    # 3. Limpieza: Eliminar fecha y columnas no necesarias
    cols_to_drop = ['fecha_prestamo'] 
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 4. Split de datos
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Identificar tipos de variables
    num_vars = X_train.select_dtypes(include=['number']).columns.tolist()
    cat_vars = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # 5. Pipeline Robusto
    processing_pipeline = Pipeline([
        ('num_imputer', MeanMedianImputer(imputation_method='median', variables=num_vars)),
        ('cat_imputer', CategoricalImputer(imputation_method='frequent', variables=cat_vars)),
        ('one_hot', OneHotEncoder(variables=cat_vars, drop_last=True)),
        ('scaler', SklearnTransformerWrapper(transformer=StandardScaler(), variables=num_vars))
    ])

    X_train_transformed = processing_pipeline.fit_transform(X_train)
    X_test_transformed = processing_pipeline.transform(X_test)

    print(f"✅ Ingeniería terminada. Procesadas {X_train_transformed.shape[1]} variables.")
    return X_train_transformed, X_test_transformed, y_train, y_test, processing_pipeline

if __name__ == "__main__":
    run_feature_engineering()