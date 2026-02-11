import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from ft_engineering import run_feature_engineering

def summarize_classification(y_true, y_pred, y_proba, model_name):
    """Métricas solicitadas en el Avance 2."""
    return {
        'Modelo': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1-Score': f1_score(y_true, y_pred, average='weighted'),
        'ROC-AUC': roc_auc_score(y_true, y_proba[:, 1]) if y_proba.shape[1] > 1 else 0
    }

def run_evaluation_pipeline():
    # 1. Preparar datos (usa la ruta por defecto ../Base_de_datos.xlsx)
    try:
        X_train, X_test, y_train, y_test, pipe = run_feature_engineering()
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Modelos
    models = [
        ('Regresión Logística', LogisticRegression(max_iter=2000)),
        ('Random Forest', RandomForestClassifier(n_estimators=150, random_state=42))
    ]

    results_list = []
    trained_models = {}

    # 3. Entrenamiento
    for name, model in models:
        print(f"🚀 Entrenando {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        results_list.append(summarize_classification(y_test, y_pred, y_proba, name))
        trained_models[name] = model

        # Matriz de Confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz: {name}')
        plt.show()

    # 4. Resultados
    df_res = pd.DataFrame(results_list)
    print("\n📊 RESULTADOS:")
    print(df_res)

    # 5. Guardar Mejor Modelo
    best_name = df_res.loc[df_res['F1-Score'].idxmax()]['Modelo']
    best_model = trained_models[best_name]
    
    final_artifact = {'pipeline': pipe, 'model': best_model, 'features': X_train.columns.tolist()}
    joblib.dump(final_artifact, 'mejor_modelo_crediticio.pkl')
    print(f"\n🏆 Ganador: {best_name}. Guardado como 'mejor_modelo_crediticio.pkl'")

if __name__ == "__main__":
    run_evaluation_pipeline()