
::: {#d7d29092-affe-4507-a752-e5bf97b7e887 .cell .markdown}
# Predicción de la Calidad del Vino. Clasificación vs regresión {#predicción-de-la-calidad-del-vino-clasificación-vs-regresión}

## Introducción

El objetivo de esta práctica es predecir la calidad del vino utilizando
técnicas de aprendizaje automático. Se utilizará el dataset Wine Quality
del repositorio de la UCLM proporcionadio para la prueba. Se
implementarán dos enfoques:

Un modelo de regresión, tratando la calidad como una variable continua.

Un modelo de clasificación, tratando la calidad como una variable
categórica. Posteriormente se evaluará el rendimiento de ambos modelos y
se compararán los resultados.

## Carga del dataset
:::

::: {#d26dabd0-6eaa-4b82-a716-1e4de666febe .cell .code}
``` python
# Libreria de python proporcionada para la prueba
!pip install ucimlrepo
```
:::

::: {#5c5e0204-ed33-4b4a-98f3-b50db84ad14b .cell .code}
``` python
# Libreria de scikit-learn necesaria para realizar la prueba referida en los imports como "sklearn"
!pip install scikit-learn
```
:::

::: {#4d6d8459-29a3-497a-8223-2dd481ff6a49 .cell .code}
``` python
from ucimlrepo import fetch_ucirepo
# Descarga del dataset
wine_quality = fetch_ucirepo(id=186)
# Datos brutos en formato Dataframe de pandas
X = wine_quality.data.features
y = wine_quality.data.targets

# Metadatos
print(wine_quality.metadata)
# Información de las variables
print(wine_quality.variables)
```
:::

::: {#0ce29411-2dac-4f87-a8e3-5c00a9a80da6 .cell .markdown}
## Elección de algoritmo

Se ha elegido **Random Forest Regressor** para la calidad como
**variable continua** debido a que el conjunto de datos posee 4898
muestras de vino blanco y 1599 muestras de vino rojo. Siendo un conjunto
de datos relativamente pequeño y obedeciendo a la [Cheat
Sheet](https://scikit-learn.org/stable/machine_learning_map.html) de
scikit learn que también está presente en las diapositivas de teoría se
ha elegido este algoritmo de regresión descartando otros como SVR debido
a su [mayor
exactitud](https://medium.com/@shiv94357/regression-algorithms-compared-which-one-is-best-for-your-ml-project-defb5089a665).

Para el algoritmo de clasificación se ha elegido **Random Forest
Classifier** para la calidad como **variable categórica** debido a la
presencia de variables continuas y a la cantidad de muestras explicadas
en la elección del anterior algoritmo. Se ha descartado SVC por la misma
razón que el algoritmo anterior.

## Evaluación de los modelos

Los modelos serán evaluados con métricas como Mean Absolute Error (MAE),
Mean Square Error (MSE) y $𝑅^2$ para regresión y confusion matrix, ROC y
F1-score, además se añadirá \"Classification Report\" de la librería
sklearn.

### Regresión
:::

::: {#4576594d-8338-47d2-b762-0cb18b9551c0 .cell .code}
``` python
# Evaluación del modelo de regresión usando random forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Crear el modelo de regresión Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
# Entrenar el modelo
model.fit(X_train, y_train.values.ravel())
# Hacer predicciones
y_pred = model.predict(X_test)
# Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)
```
:::

::: {#2e616c2b-1f72-47aa-9539-8882ea10cb67 .cell .markdown}
### Clasificación
:::

::: {#98a5f0ce-508f-4acf-8740-676d86d339be .cell .code}
``` python
# Evaluación del modelo de clasificación usando random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
# Convertir la variable objetivo en una variable categórica. El vino es tratado como de buena calidad si su calidad es superior al 6.5
y_class = (y >= 6.5).astype(int)
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
# Crear el modelo de clasificación Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Entrenar el modelo
model.fit(X_train, y_train.values.ravel())
# Hacer predicciones
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
# Evaluar el modelo
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
print("ROC AUC Score:", roc_auc)
print("F1 Score:", f1)
```
:::

::: {#8c57c28f-7907-4f16-91f3-24aa4c47b080 .cell .markdown}
## Comparación

Los resultados obtenidos son, para la **regresión**:
Mean Absolute Error (MAE): 0.4378538461538462
Mean Squared Error (MSE): 0.3713522307692308
R2 Score: 0.49718520459177784

Y, para **clasificación**:
Confusion Matrix: ((1009 39)(109 143))\*
ROC AUC Score: 0.9192888192172542
F1 Score: 0.6589861751152074

La interpretación de estos resultados indica que **el modelo de
clasificación es mucho mas preciso** para clasificar vinos de alta
calidad (AUC≈0.92) mientras que el de regresión solo puede explicar
cerca de la mitad de la varianza (R2≈0.50). Sobre el funcionamiento de
los dos aproximaciones depende en gran medida de la utilidad de las
mediciones. Si fuese necesario conocer el dato exacto de la calidad el
modelo de regresión sería más adecuado. Pero desde la perspectiva
comercial que es la mas probable sobre la que resulte beneficiosa un
modelo que identifique vinos de alta calidad de forma precisa sería mas
adecuado. Para añadir desde el punto de vista de la computación es más
facil la clasificación pero la simplificación del problema categorizando
la calidad.

\*Se ha omitido el uso de corchetes para no incomodar con el añadido de
hipervínculos

## Github

El código utilizado y este cuaderno serán almacenados en [mi
repositorio](https://github.com/errepeje/Wine_Quality_Prediction) para
la prueba de progreso

## Fuentes

La mayoría de fragmentos de código se han extraído de los ejemplos de la
librería de scikit-learn, sobretodo de
[RandomForestRegresor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html),
[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
y [Metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html)
:::
