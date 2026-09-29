import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Data
df = pd.read_csv("data/raw/dataset_practica_final.csv")

target = "is_canceled"
drop_cols = [target, "reservation_status", "reservation_status_date"]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[target]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

models = {
    "Regresión logística": LogisticRegression(max_iter=1000),
    "Árbol de decisión": DecisionTreeClassifier(random_state=42)
}

results = {}
tabla_filas = []

for name, clf in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", clf)
    ])
    
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    tabla_filas.append([
        name,
        f"{f1:.4f}".replace(".", ","),
        f"{acc:.4f}".replace(".", ","),
        f"{prec:.4f}".replace(".", ","),
        f"{rec:.4f}".replace(".", ","),
        f"{auc:.4f}".replace(".", ",")
    ])
    
    results[name] = pipe

# Crear figura con espacio extra abajo para la tabla
fig = plt.figure(figsize=(10, 9))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.4])

# Graficar Matrices y ROCs
for i, (name, pipe) in enumerate(results.items()):
    ax_cm = fig.add_subplot(gs[i, 0])
    ax_roc = fig.add_subplot(gs[i, 1])
    
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax_cm, cmap="Blues")
    ax_cm.set_title(f"Matriz de Confusión - {name}")
    
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax_roc)
    ax_roc.set_title(f"Curva ROC - {name}")

# Dibujar la tabla de métricas en la parte inferior de la imagen
ax_table = fig.add_subplot(gs[2, :])
ax_table.axis("off")

cols = ["Modelo", "F1-Score", "Accuracy", "Precisión", "Recall", "ROC-AUC"]
tabla_plot = ax_table.table(
    cellText=tabla_filas,
    colLabels=cols,
    loc="center",
    cellLoc="center"
)

tabla_plot.scale(1, 1.5)
tabla_plot.set_fontsize(10)

plt.tight_layout()
plt.show()