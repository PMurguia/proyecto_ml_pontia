import pandas as pd
import matplotlib.pyplot as plt
from pycaret.classification import *

df = pd.read_csv("data/raw/dataset_practica_final.csv")

s = setup(
    data=df,
    target="is_canceled",
    ignore_features=["reservation_status", "reservation_status_date"],
    train_size=0.8,
    fold=5,
    session_id=42,
    fix_imbalance=False,
    normalize=False,
)

# Comparativa inicial de todos los algoritmos disponibles
comparacion_inicial = compare_models(sort="AUC", n_select=3)

# Modelos por defecto de PyCaret
modelo_rf_default = create_model("rf")
modelo_xgb_default = create_model("xgboost")

# Modelos con los hiperparámetros elegidos por nosotros
modelo_rf_entrenado  = create_model("rf",n_estimators=300,min_samples_leaf=2,max_depth=None,n_jobs=-1,random_state=42)
modelo_xgb_entrenado = create_model("xgboost",n_estimators=300,learning_rate=0.05,max_depth=6,n_jobs=-1,random_state=42)

rf_tuned = tune_model(modelo_rf_default, optimize="AUC", n_iter=20)
xgb_tuned = tune_model(modelo_xgb_default, optimize="AUC", n_iter=20)

plot_model(rf_tuned, plot="confusion_matrix")
plot_model(rf_tuned, plot="auc")
plot_model(rf_tuned, plot="feature")
plot_model(rf_tuned, plot="pr")

print("Random Forest ajustado:", rf_tuned)
print("XGBoost ajustado:", xgb_tuned)

#Comparacion total con los modelos por defectos y los entrenados por nosotros
comparacion = compare_models(include=[
    modelo_rf_default,
    modelo_rf_entrenado,
    rf_tuned,
    modelo_xgb_default,
    modelo_xgb_entrenado,
    xgb_tuned,
])
