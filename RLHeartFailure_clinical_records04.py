
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import seaborn as sns
#Fuente del dataset : 

#https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data
"""""
citation: Davide Chicco, Giuseppe Jurman: 
Machine learning can predict survival of patients with heart failure
from serum creatinine and ejection fraction alone. 
BMC Medical Informatics and Decision Making 20, 16 (2020). (https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5)
"""
datos = pd.read_csv('heart_failure_clinical_records_dataset.csv')

print("++INFORMACIÓN DEL DATASE ORIGINAL++++")
print("N:", len(datos))
print(datos.info())
#Model 1. AUC74:X = datos[['ejection_fraction','serum_creatinine']]

#Model 2: AUC=84
X = datos[['age','ejection_fraction','creatinine_phosphokinase', 'ejection_fraction', 'time', 'sex']]
Y = datos['DEATH_EVENT']


# Dividir el conjunto de datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# Imputa los valores faltantes en Y_train
imputer = SimpleImputer(strategy='most_frequent')

Y_train = imputer.fit_transform(Y_train.values.reshape(-1, 1))
Y_train = Y_train.flatten()

# Imputa los valores faltantes en Y_test
Y_test = imputer.transform(Y_test.values.reshape(-1, 1))
Y_test = Y_test.flatten()

# Imputa los valores faltantes en X_train
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

# Imputa los valores faltantes en X_test
X_test = imputer.transform(X_test)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, Y_train)

# Realizar predicciones en el conjunto de prueba
Y_pred = model.predict(X_test)

# Calcular el grado de error general en los datos de prueba
error_general = 1 - metrics.accuracy_score(Y_test, Y_pred)

# Calcular el porcentaje de acierto en los datos de prueba
acierto = metrics.accuracy_score(Y_test, Y_pred)

# Calcular las métricas de matriz de confusión
confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred)
verdaderos_positivos = confusion_matrix[1, 1]
falsos_negativos = confusion_matrix[1, 0]
falsos_positivos = confusion_matrix[0, 1]
verdaderos_negativos = confusion_matrix[0, 0]

# Calcular sensibilidad y especificidad
sensibilidad = verdaderos_positivos / (verdaderos_positivos + falsos_negativos)
especificidad = verdaderos_negativos / (verdaderos_negativos + falsos_positivos)

# Imprimir resultados
print("Grado de error general en los datos de prueba:", error_general)
print("Porcentaje de acierto en los datos de prueba:", acierto)
print("Porcentaje de error para Verdaderos Positivos:", falsos_negativos / (verdaderos_positivos + falsos_negativos))
print("Porcentaje de acierto para Verdaderos Positivos:", verdaderos_positivos / (verdaderos_positivos + falsos_negativos))
print("Porcentaje de error para Falsos Negativos:", falsos_positivos / (verdaderos_negativos + falsos_positivos))
print("Porcentaje de acierto para Falsos Negativos:", verdaderos_negativos / (verdaderos_negativos + falsos_positivos))
print("Porcentaje de error para Falsos Positivos:", falsos_positivos / (verdaderos_negativos + falsos_positivos))
print("Porcentaje de acierto para Falsos Positivos:", verdaderos_negativos / (verdaderos_negativos + falsos_positivos))
print("Porcentaje de error para Verdaderos Negativos:", falsos_negativos / (verdaderos_positivos + falsos_negativos))
print("Porcentaje de acierto para Verdaderos Negativos:", verdaderos_positivos / (verdaderos_positivos + falsos_negativos))
print("Sensibilidad:", sensibilidad)
print("Especificidad:", especificidad)

# Obtener las probabilidades predichas para la clase positiva
Y_prob = model.predict_proba(X_test)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)

# Calcular el área bajo la curva ROC (AUC)
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
plt.title("Curva ROC Model 2: age,ejection_fraction,creatinine_phosphokinase, ejection_fraction, time, sex")
plt.legend(loc='lower right')
plt.show()


# Mostrar gráfico de distribución de frecuencias para las variables X
plt.figure(figsize=(12, 6))
plt.suptitle('Distribución de Frecuencias de Variables X', fontsize=16)
for i, column in enumerate(X.columns):
    plt.subplot(2, 3, i + 1)
    sns.histplot(X[column], kde=True, color='skyblue', bins=20)
    plt.title(f'Distribución {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

