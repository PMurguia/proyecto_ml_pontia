import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline          # CORREGIDO: era "Pipelines"
from sklearn.base import clone                 # AÑADIDO: para independizar los pipelines
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score)

df = pd.read_csv("data/raw/dataset_practica_final.csv")

y_heart = df["is_canceled"]
X_heart = df.drop(columns=["is_canceled", "reservation_status", "reservation_status_date"]) #Elimino reservation_status y reservation_status_date porque se registran después de que la reserva se cancela o se completa. en el momento de predecir no estarían disponibles y el modelo aprendería a leer la respuesta

meses = {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6,
         "July":7, "August":8, "September":9, "October":10, "November":11, "December":12}

df["arrival_month_num"] = df["arrival_date_month"].map(meses)
df["arrival_date"] = pd.to_datetime(dict(year=df["arrival_date_year"],month=df["arrival_month_num"],day=df["arrival_date_day_of_month"])) # Unifico dia,mes y año y convierto a entero los meses del año

cat_cols = X_heart.select_dtypes(include=["object", "str"]).columns.tolist()
num_cols = [c for c in X_heart.columns if c not in cat_cols]

# Pipeline:
preproceso = ColumnTransformer([
    ("cat", Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("codificar", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), cat_cols),
    ("num", SimpleImputer(strategy="median"), num_cols),
])

modelo_rfc = Pipeline([
    ("preproceso", clone(preproceso)),
    ("clasificador", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                            n_jobs=-1, random_state=42)),
])

modelo_xgb = Pipeline([
    ("preproceso", clone(preproceso)),
    ("clasificador", XGBClassifier(n_estimators=300,learning_rate=0.05,
                                   max_depth=6,n_jobs=-1,random_state=42)),
])

# Entrenamiento del modelo Random Forest

X_train, X_test, y_train, y_test = train_test_split(X_heart, y_heart, test_size=0.2, stratify=y_heart, random_state=42)

modelo_rfc.fit(X_train,y_train)
print(modelo_rfc.score(X_test,y_test))

y_pred = modelo_rfc.predict(X_test)

# Obtención de las métricas de evaluación
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {acc:.2%}")
print(f"Precisión: {prec:.2%}")
print(f"Recall:    {rec:.2%}")
print(f"F1-Score:  {f1:.2%}")
print(f"ROC-AUC: {roc_auc_score(y_test, modelo_rfc.predict_proba(X_test)[:,1]):.4f}")

# Comparamos valores reales con los valores del dummy
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("Dummy accuracy:", round(dummy.score(X_test, y_test), 4)) 

# Matriz de confusion y Curva ROC
ConfusionMatrixDisplay.from_estimator(
    modelo_rfc, X_test, y_test,
    display_labels=["No cancela", "Cancela"],
    cmap="Blues",
    normalize=None)
plt.title("Matriz de confusión - Random Forest")
plt.show()

RocCurveDisplay.from_estimator(modelo_rfc, X_test, y_test, name="Random Forest")
plt.plot([0,1], [0,1], "k--", label="Azar (AUC=0.5)")
plt.title("Curva ROC")
plt.legend()
plt.show()

modelo_xgb.fit(X_train,y_train)
print(modelo_xgb.score(X_test,y_test))

y_pred = modelo_xgb.predict(X_test)

# Obtención de las métricas de evaluación
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {acc:.2%}")
print(f"Precisión: {prec:.2%}")
print(f"Recall:    {rec:.2%}")
print(f"F1-Score:  {f1:.2%}")
print(f"ROC-AUC: {roc_auc_score(y_test, modelo_xgb.predict_proba(X_test)[:,1]):.4f}")

# Matriz de confusion y Curva ROC
ConfusionMatrixDisplay.from_estimator(
    modelo_xgb, X_test, y_test,
    display_labels=["No cancela", "Cancela"],
    cmap="Blues",
    normalize=None)
plt.title("Matriz de confusión - Gradiant Boosting")
plt.show()

RocCurveDisplay.from_estimator(modelo_xgb, X_test, y_test, name="Gradiant Boosting")
plt.plot([0,1], [0,1], "k--", label="Azar (AUC=0.5)")
plt.title("Curva ROC")
plt.legend()
plt.show()

# Curva ROC comparativa de los dos modelos en un mismo gráfico
ax = plt.gca()
RocCurveDisplay.from_estimator(modelo_rfc, X_test, y_test, ax=ax, name="Random Forest")
RocCurveDisplay.from_estimator(modelo_xgb, X_test, y_test, ax=ax, name="Gradiant Boosting")
plt.plot([0,1], [0,1], "k--", label="Azar (AUC=0.5)")
plt.title("Curva ROC comparativa")
plt.legend()
plt.show()
