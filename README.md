# Predicción de cancelaciones hoteleras

Sistema de clasificación binaria que implementa, evalúa y compara cinco
algoritmos de aprendizaje supervisado sobre un dataset real de reservas
hoteleras.

Práctica final del módulo de Machine Learning y Deep Learning — Máster en
Inteligencia Artificial, Cloud Computing y DevOps (PontIA).

---

## 1. Autores y roles

| Integrante | Archivo | Responsabilidad |
| ---------- | ------- | --------------- |
| **Jorge Pozo Galán** | `regresion_logistica_y_arbol_decisiones.py` | Regresión logística y árbol de decisión. Preprocesado con one-hot encoding y escalado. Justificación de F1-Score como métrica principal. Tabla comparativa integrada en la figura. |
| **Peio Murguia** | `red_neuronal.py` | Red neuronal multicapa con Keras/TensorFlow. Arquitectura, regularización con dropout y early stopping. Exportación de métricas a CSV. |
| **Pablo Calderón** | `RFXB - Models.py`, `RFXB - Pycaret.py` | Random Forest y XGBoost. Detección de la fuga de información. Pipeline con codificación ordinal. Comparativa con AutoML (PyCaret). |

Cada integrante ha desarrollado sus modelos de forma autónoma, acordando
conjuntamente la partición de datos y el tratamiento de la fuga de información
para que los resultados fueran comparables.

---

## 2. Descripción del problema y de los datos

Predecir si una reserva de hotel terminará cancelándose (`is_canceled = 1`) o
no (`is_canceled = 0`). Anticipar cancelaciones permite a un hotel gestionar el
overbooking, ajustar precios y planificar personal y compras.

**Dataset**: 119.390 reservas de dos hoteles portugueses (urbano y resort)
entre julio de 2015 y agosto de 2017, con 31 variables sobre el cliente, el
comportamiento de reserva y el canal de distribución.

**Balance de clases**: 62,96 % no canceladas / 37,04 % canceladas.

### Fuga de información

Dos columnas filtran la variable objetivo y se eliminan en **los tres scripts**
antes de entrenar:

| Columna | Motivo |
| ------- | ------ |
| `reservation_status` | Es el target reetiquetado. Correspondencia 1 a 1: `Check-Out` → 0 (75.166), `Canceled` → 1 (43.017), `No-Show` → 1 (1.207). |
| `reservation_status_date` | Fecha del desenlace. Para una cancelación es anterior a la llegada, así que la diferencia con la fecha de llegada delata el target. |

Sin eliminarlas, cualquier modelo alcanza un 100 % de acierto y no aprende
nada. Fue la primera decisión del proyecto y condiciona todo lo demás.

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
├── docs/
│   └── informe_final.md
├── src/
│   ├── regresion_logistica_y_arbol_decisiones.py   # Jorge Pozo
│   ├── red_neuronal.py                             # Peio Murguia
│   ├── RFXB - Models.py                            # Pablo Calderón
│   └── RFXB - Pycaret.py                           # Pablo Calderón (AutoML)
├── notebooks/                                      # EDA y pruebas
└── outputs/                                        # Gráficos y métricas generados
```

---

## 4. Instalación y ejecución

Requiere **Python 3.11**.

```bash
conda create -n practica-ml python=3.11 -y
conda activate practica-ml
pip install -r requirements.txt
```

Cada script es independiente y se ejecuta desde la raíz del repositorio:

```bash
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

Todos los scripts leen el CSV desde `data/raw/dataset_practica_final.csv`.

---

## 5. Diseño del sistema

```
CSV crudo
   │
   ├─ Eliminar columnas con fuga
   ├─ Separar X / y
   ├─ train_test_split(test_size=0.2, stratify=y, random_state=42)
   │
   └─ Pipeline([ColumnTransformer, modelo])
            ├─ numéricas:   SimpleImputer(mediana) [+ StandardScaler]
            └─ categóricas: SimpleImputer(moda) + encoder
                  │
                  └─► métricas + matriz de confusión + curva ROC
```

Todos los modelos se envuelven en un `Pipeline` de scikit-learn, de modo que el
preprocesado se ajusta **únicamente** con los datos de entrenamiento: codificar
antes de dividir dejaría que el encoder viese el conjunto de test.

### Preprocesado adaptado a cada familia de modelos

| Modelo | Codificación | Escalado | Motivo |
| ------ | ------------ | -------- | ------ |
| Regresión logística | One-hot | Sí | Un modelo lineal interpretaría un código ordinal como magnitud, y necesita variables en escalas comparables para converger |
| Árbol de decisión | One-hot | Sí | Comparte pipeline con la regresión logística |
| Red neuronal | One-hot | Sí | El descenso de gradiente requiere entradas normalizadas |
| Random Forest, XGBoost | Ordinal | No | Con `country` (177 valores) y `agent` (333), el one-hot generaría ~500 columnas dispersas. Un árbol reagrupa categorías cortando donde quiera, así que el orden arbitrario no le perjudica |

La partición es idéntica en los tres scripts (`test_size=0.2`, `stratify=y`,
`random_state=42`), de modo que todos los modelos se evalúan sobre las mismas
23.878 reservas.

---

## 6. Métricas de evaluación

**F1-Score** se emplea como métrica principal común. Con un 37 % de positivos,
la accuracy engaña: un clasificador que siempre responda "no cancela" ya
acierta el 62,96 % sin haber aprendido nada. F1 equilibra precisión y recall
sobre la clase minoritaria, que es la que interesa detectar.

**ROC-AUC** se reporta como métrica secundaria. Su ventaja es que no depende
del umbral de decisión: las demás métricas se calculan tras aplicar un corte de
0,5 sobre la probabilidad, y ese 0,5 es una convención, no una propiedad del
modelo.

También se reportan accuracy, precisión y recall. El **recall** tiene lectura
directa de negocio: cada cancelación no detectada es una habitación que queda
vacía sin haberse revendido.

---

## 7. Resultados

Partición 80/20 estratificada, semilla 42, hiperparámetros por defecto de cada
script.

| Modelo | F1-Score | Accuracy | Precisión | Recall | ROC-AUC |
| ------ | -------- | -------- | --------- | ------ | ------- |
| Regresión logística | 0,7302 | 0,8182 | 0,8112 | 0,6639 | 0,8961 |
| Árbol de decisión | 0,8111 | 0,8593 | 0,8066 | 0,8157 | 0,8528 |
| Red neuronal (Keras) | 0,8123 | 0,8697 | 0,8708 | 0,7612 | 0,9454 |
| XGBoost | 0,8229 | 0,8739 | 0,8577 | 0,7908 | 0,9490 |
| Random Forest | 0,8486 | 0,8929 | 0,8904 | 0,8105 | 0,9595 |
| *Dummy (referencia)* | 0,0000 | 0,6296 | 0,0000 | 0,0000 | 0,5000 |

Los cinco modelos superan con holgura al clasificador trivial, lo que confirma
que las métricas reflejan aprendizaje real y no el desbalance de clases.

### Modelo seleccionado

Según F1-Score sobre esta partición, el **Random Forest** obtiene el valor más
alto (0,8486) y es el que mejor equilibra precisión y recall, además de liderar
también en ROC-AUC (0,9595). Se selecciona como modelo final del sistema.

### Cómo leer esta tabla

Las cifras corresponden a **una única partición**. Tres matices:

1. **No hay validación cruzada.** Diferencias por debajo de unos 2 puntos de F1
   podrían deberse al azar de la partición concreta. Solo un `StratifiedKFold`
   con su desviación típica permitiría distinguirlas del ruido.
2. **El preprocesado no es idéntico.** Cada familia de modelos recibe el
   tratamiento que su naturaleza requiere, lo que significa que no ven la misma
   representación de los datos.
3. **Los hiperparámetros no están optimizados por igual.** Ningún modelo de esta
   tabla ha pasado por una búsqueda sistemática.

### Un resultado que merece comentario

El árbol de decisión obtiene un F1 alto (0,8111) pero el **ROC-AUC más bajo de
todos** (0,8528), por debajo incluso de la regresión logística. No es una
contradicción: un árbol sin podar produce probabilidades muy próximas a 0 o a 1,
así que su curva ROC tiene pocos escalones y encierra menos área. Acierta bien
con el umbral de 0,5, pero ordena peor las reservas por riesgo. Es un buen
ejemplo de por qué conviene mirar más de una métrica.

### Matriz de confusión del modelo seleccionado

|  | Predicho: no cancela | Predicho: cancela |
| --- | --- | --- |
| **Real: no cancela** | 14.151 | 882 |
| **Real: cancela** | 1.676 | 7.169 |

Los dos errores tienen costes distintos: 882 falsos positivos (predecimos
cancelación y el cliente viene, riesgo de overbooking) frente a 1.676 falsos
negativos (predecimos que viene y cancela, habitación vacía). Todos los modelos
se evalúan con umbral 0,5, que no optimiza esa asimetría.

### Gráficos

Cada script genera su matriz de confusión y su curva ROC en `outputs/`.
`RFXB - Models.py` incluye además una curva ROC comparativa de Random Forest y
XGBoost en un mismo gráfico, y `regresion_logistica_y_arbol_decisiones.py`
integra la tabla de métricas dentro de la figura.

---

## 8. Conclusiones

1. **La detección de la fuga de información fue la decisión determinante.**
   Incluir `reservation_status` produce un AUC de 1,0 y un modelo sin ningún
   valor. Ninguna herramienta lo detecta automáticamente: PyCaret, sin
   `ignore_features`, devuelve resultados perfectos y falsos.

2. **Los cinco algoritmos aprenden señal real** y quedan muy por encima del
   clasificador trivial, lo que confirma que el problema es predecible con las
   variables disponibles y que el preprocesado es correcto.

3. **Los modelos no lineales captan mejor la estructura del problema.** La
   regresión logística es el único que se queda claramente por detrás en F1
   (0,7302), con un recall de 0,6639 frente al 0,81 del Random Forest. Sugiere
   que las cancelaciones dependen de interacciones entre variables
   (`lead_time` × `deposit_type` × `market_segment`) que un modelo lineal no
   puede representar.

4. **Cada familia aporta algo distinto.** La regresión logística es
   interpretable y rápida; el árbol produce reglas legibles para un gestor
   hotelero; la red neuronal capta interacciones sin especificarlas; los
   conjuntos de árboles son robustos frente a outliers y alta cardinalidad. En
   un entorno real la elección dependería también del coste de cómputo y de la
   necesidad de explicar las decisiones.

5. **El umbral de decisión importa tanto como el algoritmo.** Moverlo de 0,5 a
   0,35 aumenta el recall a costa de la precisión, sin reentrenar nada. Para un
   hotel, una habitación vacía suele costar más que ofrecer una plaza que no
   hacía falta.

6. **PyCaret reproduce resultados coherentes con el pipeline manual**, lo que
   valida el preprocesado. Su aportación es la rapidez de prototipado; el
   criterio sobre qué variables deben entrar sigue siendo del analista.

---

## 9. Limitaciones y mejoras

- **Sin validación cruzada**: es la limitación más relevante. Todas las cifras
  provienen de una partición única.
- **Duplicados**: 31.994 filas exactamente iguales (26,8 %). Se conservan por
  considerarse reservas de grupo reales, pero inflan el resultado del split
  aleatorio.
- **Validación temporal pendiente**: entrenar con 2015-2016 y validar con 2017
  baja el AUC del Random Forest de 0,96 a 0,88. Ese es el rendimiento realista
  en producción, y es previsible que afecte de forma similar al resto.
- **Variables sospechosas**: de las 7.416 reservas que piden parking, ninguna
  cancela. Una separación perfecta sugiere que el dato se registra en
  recepción, es decir, después del momento de predicción.
- **`deposit_type = "Non Refund"` cancela el 99,4 %** de las veces. Es
  información legítima, pero domina cualquier modelo y tapa al resto.
- **La red neuronal no fija la semilla de TensorFlow**, así que sus métricas
  varían ligeramente entre ejecuciones.
- **Mejoras propuestas**: validación cruzada con desviación típica, búsqueda
  sistemática de hiperparámetros, target encoding para variables de alta
  cardinalidad, calibración de probabilidades, optimización del umbral por coste
  económico y exposición del modelo vía API REST.
