# Predicción de cancelaciones hoteleras

Sistema de clasificación binaria que implementa, evalúa y compara cinco
algoritmos de aprendizaje supervisado sobre un dataset real de reservas
hoteleras.

Práctica final del módulo de Machine Learning y Deep Learning.

---

## 1. Autores y roles

| Integrante | Archivo | Responsabilidad |
| ---------- | ------- | --------------- |
| **Jorge Pozo Galán** | `regresion_logistica_y_arbol_decisiones.py` | Regresión logística y árbol de decisión. Preprocesado con one-hot encoding y escalado. Justificación de F1-Score como métrica principal. |
| **Peio Murguia** | `red_neuronal.py` | Red neuronal multicapa con Keras/TensorFlow. Arquitectura, regularización con dropout y early stopping. |
| **Pablo Calderón** | `RFXB - Models.py`, `RFXB - Pycaret.py` | Random Forest y XGBoost. Detección de la fuga de información. Pipeline con codificación ordinal. Comparativa con AutoML (PyCaret). |


---

## 2. Descripción del problema y de los datos

Predecir si una reserva de hotel terminará cancelándose (`is_canceled = 1`) o
no (`is_canceled = 0`). Anticipar cancelaciones permite a un hotel gestionar el
overbooking, ajustar precios y planificar personal.

**Dataset**: 119.390 reservas de dos hoteles portugueses (urbano y resort)
entre 2015 y 2017, con 31 variables sobre el cliente, el comportamiento de
reserva y el canal de distribución.

**Balance de clases**: 62,96 % no canceladas / 37,04 % canceladas.

### Fuga de información

Dos columnas filtran la variable objetivo y se eliminan en **los tres scripts**
antes de entrenar:

| Columna | Motivo |
| ------- | ------ |
| `reservation_status` | Es el target reetiquetado. Correspondencia 1 a 1: `Check-Out` → 0 (75.166), `Canceled` → 1 (43.017), `No-Show` → 1 (1.207). |
| `reservation_status_date` | Fecha del desenlace. Para una cancelación es anterior a la llegada, así que la diferencia con la fecha de llegada delata el target. |

Sin eliminarlas, cualquier modelo alcanza un 100 % de acierto y no aprende
nada. Esta fue la primera decisión del proyecto y condiciona todo lo demás.

---

## 3. Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw/
│       └── dataset_practica_final.csv
├── src/
│   ├── regresion_logistica_y_arbol_decisiones.py   # Jorge Pozo
│   ├── red_neuronal.py                             # Peio Murguia
│   ├── RFXB - Models.py                            # Pablo Calderón
│   └── RFXB - Pycaret.py                           # Pablo Calderón (AutoML)
├── notebooks/                                      # EDA y pruebas
└── outputs/                                        # Gráficos generados
```

---

## 4. Instalación y ejecución

Requiere **Python 3.11**.

```bash
conda create -n practica-ml python=3.11 -y
conda activate practica-ml
pip install -r requirements.txt

# Cada script es independiente
python "src/RFXB - Models.py"
python src/regresion_logistica_y_arbol_decisiones.py
python src/red_neuronal.py
```

> **PyCaret aparte.** `RFXB - Pycaret.py` necesita un entorno propio: fija
> versiones de scikit-learn y pandas que entran en conflicto con TensorFlow y
> XGBoost.
>
> ```bash
> conda create -n pycaret python=3.11 -y
> conda activate pycaret
> pip install pycaret
> python "src/RFXB - Pycaret.py"
> ```

Todos los scripts leen el CSV desde `data/raw/dataset_practica_final.csv` y
deben ejecutarse desde la raíz del repositorio.

---

## 5. Diseño del sistema

```
CSV
   │
   ├─ Eliminar columnas con fuga
   ├─ Separar X / y
   ├─ train_test_split(test_size=0.2, stratify=y, random_state=42)
   │
   └─ Pipeline([ColumnTransformer, modelo])
            ├─ numéricas:   SimpleImputer(mediana) [+ StandardScaler]
            └─ categóricas: SimpleImputer + encoder
                  │
                  └─► métricas + matriz de confusión + curva ROC
```

Todos los modelos se envuelven en un `Pipeline` de scikit-learn, de modo que el
preprocesado se ajusta **únicamente** con los datos de entrenamiento: codificar
antes de dividir dejaría que el encoder viese el conjunto de test.

### Preprocesado adaptado a cada familia de modelos

Cada algoritmo recibe el tratamiento que su naturaleza requiere. No es una
inconsistencia, sino una decisión deliberada:

| Modelo | Codificación | Escalado | Motivo |
| ------ | ------------ | -------- | ------ |
| Regresión logística | One-hot | Sí | Un modelo lineal interpretaría un código ordinal como una magnitud, y necesita variables en escalas comparables para converger |
| Árbol de decisión | One-hot | Sí | Comparte pipeline con la regresión logística |
| Red neuronal | One-hot | Sí | El descenso de gradiente requiere entradas normalizadas |
| Random Forest, XGBoost | Ordinal | No | Con `country` (177 valores) y `agent` (333), el one-hot generaría ~500 columnas dispersas. Un árbol reagrupa categorías cortando donde quiera, así que el orden arbitrario no le perjudica |

La partición es idéntica en los tres scripts (`test_size=0.2`, `stratify=y`,
`random_state=42`), lo que garantiza que todos los modelos se evalúan sobre las
mismas reservas.

---

## 6. Métricas de evaluación

**F1-Score** se emplea como métrica principal común. Con un 37 % de positivos,
la accuracy engaña: un clasificador que siempre responda "no cancela" ya
acierta el 63 % sin haber aprendido nada. F1 equilibra precisión y recall sobre
la clase minoritaria, que es la que interesa detectar.

**ROC-AUC** se reporta como métrica secundaria donde está disponible. Su
ventaja es que no depende del umbral de decisión: las demás métricas se
calculan tras aplicar un corte de 0,5 sobre la probabilidad, y ese 0,5 es una
convención, no una propiedad del modelo.

También se reportan accuracy, precisión y recall. El **recall** tiene lectura
directa de negocio: cada cancelación no detectada es una habitación que queda
vacía sin haberse revendido.

---

## 7. Resultados

Partición 80/20 estratificada, semilla 42, hiperparámetros por defecto de cada
script.

| Modelo | F1-Score | Accuracy | Precisión | Recall | ROC-AUC |
| ------ | -------- | -------- | --------- | ------ | ------- |
| Regresión logística | 0,7302 | — | — | — | — |
| Árbol de decisión | 0,8111 | — | — | — | — |
| Red neuronal (Keras) | — | — | — | — | — |
| Random Forest | 0,8486 | 89,29 % | 89,04 % | 81,05 % | 0,9595 |
| XGBoost | 0,8229 | 87,39 % | 85,77 % | 79,08 % | 0,9490 |
| *Dummy (referencia)* | 0,0000 | 62,96 % | 0 % | 0 % | 0,5000 |



### Matriz de confusión (Random Forest, a modo de ejemplo)

|  | Predicho: no cancela | Predicho: cancela |
| --- | --- | --- |
| **Real: no cancela** | 14.140 | 862 |
| **Real: cancela** | 1.651 | 7.189 |

Los dos errores tienen costes distintos: 862 falsos positivos (predecimos
cancelación y el cliente viene, riesgo de overbooking) frente a 1.651 falsos
negativos (predecimos que viene y cancela, habitación vacía).

### Gráficos

Cada script genera su matriz de confusión y su curva ROC.

---

## 8. Conclusiones

1. **La detección de la fuga de información fue la decisión determinante del
   proyecto.** Incluir `reservation_status` produce un AUC de 1,0 y un modelo
   sin ningún valor. Ninguna herramienta lo detecta automáticamente: PyCaret,
   sin `ignore_features`, devuelve resultados perfectos y falsos.

2. **Los cinco algoritmos aprenden señal real.** Todos quedan muy por encima
   del clasificador trivial, lo que confirma que el problema es predecible a
   partir de las variables disponibles y que el preprocesado es correcto.

3. **Cada familia de modelos aporta algo distinto.** La regresión logística es
   interpretable y rápida; el árbol de decisión produce reglas que un gestor
   hotelero puede leer; la red neuronal capta interacciones complejas sin
   especificarlas; los conjuntos de árboles son robustos frente a outliers y
   variables de alta cardinalidad.

4. **El umbral de decisión importa tanto como el algoritmo.** Todos los
   modelos se evalúan con un corte de 0,5. Moverlo a 0,35 aumenta el recall a
   costa de la precisión, sin reentrenar nada. Para un hotel, una habitación
   vacía suele costar más que ofrecer una plaza que no hacía falta.

---

## 9. Limitaciones y mejoras

- **Duplicados**: el dataset contiene ~32.000 filas exactamente iguales
  (26,8 %). Se conservan por considerarse reservas de grupo reales, pero
  inflan el resultado del split aleatorio.
- **Validación temporal pendiente**: entrenar con 2015-2016 y validar con 2017
  baja el AUC del Random Forest de 0,96 a 0,88. Ese es el rendimiento realista
  en producción, y es previsible que afecte de forma similar al resto de
  modelos.
- **Variables sospechosas**: de las 7.417 reservas que piden parking, ninguna
  cancela. Una separación perfecta sugiere que el dato se registra en
  recepción, es decir, después del momento de predicción.
- **`deposit_type = "Non Refund"` cancela el 99,4 %** de las veces. Es
  información legítima, pero domina cualquier modelo y tapa al resto de
  variables.
- **Mejoras propuestas**: validación cruzada con desviación típica, búsqueda
  sistemática de hiperparámetros para los cinco modelos, target encoding para
  variables de alta cardinalidad, calibración de probabilidades, optimización
  del umbral por coste económico y exposición del modelo vía API REST.
