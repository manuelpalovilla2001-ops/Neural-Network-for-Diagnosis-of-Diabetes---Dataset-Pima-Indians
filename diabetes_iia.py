import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score)
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass

# =====================================================================
# LECTURA DE DATOS
# =====================================================================
df = pd.read_csv('diabetes.csv')

print("--- Info del dataset ---")
print(df.head())
print(f"\nDimensiones del dataset: {df.shape}") # Filas y Columnas
print(f"\nDistribución de clases (Outcome):\n{df['Outcome'].value_counts()}")

# Detección y corrección de ceros biológicamente imposibles
columnas_sin_cero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\n--- CEROS (valores imposibles) DETECTADOS ANTES DEL REEMPLAZO ---")
print((df[columnas_sin_cero] == 0).sum())

for col in columnas_sin_cero:
    df[col] = df[col].replace(0, np.nan) # Not a Number
    df[col] = df[col].fillna(df[col].median())

X = df.drop('Outcome', axis=1).values
Y = df['Outcome'].values

print(f"\nTotal de pacientes cargados: {len(X)}")
print(f"Cantidad de variables médicas por paciente: {X.shape[1]}")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=SEED, stratify=Y
) # stratify=Y para mantener la misma proporción sano/enfermo en test y train

# Las variables del dataset tienen escalas muy distintas, las estandarizo
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

# =====================================================================
# ARQUITECTURA DEL MODELO
# =====================================================================
def construir_modelo(input_dim=8, learning_rate=0.001):
    modelo = Sequential([
        Dense(16, activation='relu', input_shape=(input_dim,),
              kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return modelo

modelo = construir_modelo()
#print("\n--- RESUMEN DE LA ARQUITECTURA ---")
#modelo.summary()

# =====================================================================
# ENTRENAMIENTO
# =====================================================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# Ajuste fino, reduzco el lr al acercarme a un buen resultado
reducir_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

print("\n--- INICIANDO ENTRENAMIENTO ---")
historial = modelo.fit(
    X_train_escalado, Y_train,
    epochs=1000,
    batch_size=32,
    validation_data=(X_test_escalado, Y_test),
    callbacks=[early_stop, reducir_lr],
    verbose=1
)
print("Entrenamiento completado")

# =====================================================================
# EVALUACION
# =====================================================================
UMBRAL = 0.30

predicciones_porcentaje = modelo.predict(X_test_escalado)
Y_prediccion = (predicciones_porcentaje > UMBRAL).astype(int)

accuracy = accuracy_score(Y_test, Y_prediccion)
precision = precision_score(Y_test, Y_prediccion)
recall = recall_score(Y_test, Y_prediccion)
f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

loss, _, auc, _, _ = modelo.evaluate(
    X_test_escalado, Y_test, verbose=0
)

print("\n--- METRICAS DEL CONJUNTO DE PRUEBA ---")
print(f"  Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)") # % de diagnosticos correctos
print(f"  AUC-ROC   : {auc:.4f}")
print(f"  Precision : {precision:.4f}") # De los diabeticos (prediccion) cuantos realmente lo son ?
print(f"  Recall    : {recall:.4f}") # De lo diabeticos totales cuantos se detectaron ?
print(f"  F1-Score  : {f1:.4f}") # Media Armonica entre Precision y Recall

print("\n--- REPORTE COMPLETO DE CLASIFICACION ---")
print(classification_report(Y_test, Y_prediccion,
                             target_names=['Sano (0)', 'Diabetes (1)']))

# =====================================================================
# GRAFICOS
# =====================================================================

# Grafico 1: Matriz de Confusión
plt.figure(figsize=(5, 4))
matriz = confusion_matrix(Y_test, Y_prediccion)
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Sano (0)', 'Diabetes (1)'],
            yticklabels=['Sano (0)', 'Diabetes (1)'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=300)
plt.close()

# Grafico 2: Evolución de Exactitud
plt.figure(figsize=(6, 4))
plt.plot(historial.history['accuracy'], label='Entrenamiento', color='blue')
plt.plot(historial.history['val_accuracy'], label='Validación', color='green')
plt.title('Evolución de la Exactitud')
plt.xlabel('Épocas')
plt.ylabel('Exactitud')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('evolucion_exactitud.png', dpi=300)
plt.close()

# Grafico 3: Evolución del Error
plt.figure(figsize=(6, 4))
plt.plot(historial.history['loss'], label='Entrenamiento', color='blue')
plt.plot(historial.history['val_loss'], label='Validación', color='red')
plt.title('Evolución de la Pérdida (Crossentropy)')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('evolucion_perdida.png', dpi=300)
plt.close()

# Grafico 4: Curva ROC
plt.figure(figsize=(5, 4))
fpr, tpr, _ = roc_curve(Y_test, predicciones_porcentaje)
roc_auc = roc_auc_score(Y_test, predicciones_porcentaje)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.title('Curva ROC')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('curva_roc.png', dpi=300)
plt.close()

# Grafico 5: AUC durante el entrenamiento
plt.figure(figsize=(6, 4))
plt.plot(historial.history['auc'], label='Entrenamiento', color='purple')
plt.plot(historial.history['val_auc'], label='Validación', color='orange')
plt.title('Evolución del AUC')
plt.xlabel('Épocas')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('evolucion_auc.png', dpi=300)
plt.close()

# =====================================================================
# IMPORTANCIA DE VARIABLES
# =====================================================================
variables = df.drop('Outcome', axis=1).columns.tolist()
loss_inicial = modelo.evaluate(X_test_escalado, Y_test, verbose=0)[0]
importancias = []

for i in range(X_test_escalado.shape[1]):
    X_permutado = X_test_escalado.copy()
    np.random.shuffle(X_permutado[:, i])
    loss_permutado = modelo.evaluate(X_permutado, Y_test, verbose=0)[0]
    importancias.append(loss_permutado - loss_inicial)

plt.figure(figsize=(8, 5))
indices_ord = np.argsort(importancias)[::-1]
plt.bar(range(len(variables)),
        [importancias[i] for i in indices_ord],
        color='steelblue')
plt.xticks(range(len(variables)),
           [variables[i] for i in indices_ord], rotation=45, ha='right')
plt.title('Importancia de Variables (Por permutación)')
plt.xlabel('Variable')
plt.ylabel('Incremento en pérdida al permutar')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('importancia_variables.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n--- IMPORTANCIA DE VARIABLES (de mayor a menor) ---")
for i in indices_ord:
    print(f"  {variables[i]:<20}: {importancias[i]:.5f}")
