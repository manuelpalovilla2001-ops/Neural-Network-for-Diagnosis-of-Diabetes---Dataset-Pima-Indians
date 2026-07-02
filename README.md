# Diagnóstico de Diabetes mediante Redes Neuronales Artificiales

Proyecto final de la materia **Introducción a la Inteligencia Artificial**  
Departamento de Ingeniería Electrónica — UTN FRBA

---

## Descripción

Este proyecto entrena un Perceptrón Multicapa (MLP) para predecir si una paciente tiene diabetes a partir de variables clínicas, usando el dataset *Pima Indians Diabetes Database*. El foco estuvo en aplicar buenas prácticas de preprocesamiento, regularización y evaluación, y en verificar que los patrones aprendidos por la red tienen coherencia médica.

El trabajo se acompaña de un [paper académico](IIA.pdf) escrito en formato IEEE.

---

## Dataset

**Pima Indians Diabetes Database** — [Fuente: Kaggle / UCI ML Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

| Característica | Valor |
|---|---|
| Pacientes | 768 mujeres, edad ≥ 21 años |
| Variables de entrada | 8 (glucosa, insulina, IMC, presión arterial, etc.) |
| Variable objetivo | `Outcome` — 0: sana, 1: diabetes |
| Distribución de clases | 65% sanas / 35% con diabetes |

### Problema de calidad de datos
Varias columnas contienen ceros biológicamente imposibles (glucosa = 0, IMC = 0, etc.), que en realidad son datos faltantes mal codificados. Se reemplazaron por la **mediana** de cada columna antes del entrenamiento.

---

## Arquitectura del modelo

```
Entrada (8) → Dense(16, ReLU) → BN → Dropout(0.3)
            → Dense(32, ReLU) → BN → Dropout(0.3)
            → Dense(16, ReLU) → BN → Dropout(0.2)
            → Dense(1, Sigmoid)
```

Cada capa oculta aplica regularización L2 (λ = 0.001), Batch Normalization y Dropout para controlar el sobreajuste.

---

## Resultados

Evaluado sobre 154 pacientes del conjunto de prueba (20% del dataset, partición estratificada):

| Métrica | Umbral 0.50 | Umbral 0.30 |
|---|---|---|
| Accuracy | 72.73% | — |
| AUC-ROC | 0.786 | 0.786 |
| Precision | 62.00% | — |
| Recall | 57.41% | — |
| F1-Score | 59.62% | — |

> El AUC-ROC no varía con el umbral porque mide el poder discriminativo del modelo en todos los umbrales posibles a la vez.

### Importancia de variables (permutación)

El análisis confirmó que la glucosa es la variable más determinante (incremento de pérdida: 0.133), seguida del historial genético (0.055) y el IMC (0.023). Esto es consistente con la literatura médica sobre factores de riesgo de la diabetes.

---

## Estructura del repositorio

```
Neural-Network-for-Diagnosis-of-Diabetes---Dataset-Pima-Indians/
│
├── diabetes_iia.py          # Código principal
├── diabetes.csv             # Dataset
├── requirements.txt         # Dependencias
├── IIA.pdf                  # Paper académico (formato IEEE)
│
└── resultados/
    ├── matriz_confusion.png
    ├── curva_roc.png
    ├── evolucion_exactitud.png
    ├── evolucion_perdida.png
    ├── evolucion_auc.png
    └── importancia_variables.png
```

---

## Instalación y uso

```bash
# 1. Clonar el repositorio
git clone https://github.com/manuelpalovilla2001-ops/Neural-Network-for-Diagnosis-of-Diabetes---Dataset-Pima-Indians
cd diabetes-neural-network

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar
python diabetes_iia.py
```

El script genera automáticamente todos los gráficos en la carpeta raíz.

---

## Decisiones de diseño relevantes

**Umbral de decisión ajustable:** la variable `UMBRAL` al inicio de la sección de evaluación permite cambiar el punto de corte sin re-entrenar el modelo. Bajar el umbral aumenta el recall (se detectan más casos de diabetes) a costa de más falsos positivos.

**Reproducibilidad:** se fijan semillas en Python, NumPy y TensorFlow para que los resultados sean idénticos en cada ejecución.

**División estratificada:** `stratify=Y` en `train_test_split` garantiza que la proporción de clases sea la misma en entrenamiento y prueba.

**Sin data leakage:** el `StandardScaler` se ajusta únicamente con los datos de entrenamiento y luego se aplica al conjunto de prueba.

---

## Limitaciones

- Inicialmente el recall con umbral 0.5 fue del 57.41%, lo que significa que el modelo no detecta aproximadamente 4 de cada 10 pacientes con diabetes. Para un uso clínico ideal se ajusto a un umbral del 0.3 obteniendo resultados en el recall significativamente superiores.
- El conjunto de entrenamiento y validación son el mismo subconjunto (el 20% de prueba se usó como `validation_data` durante el entrenamiento). Para investigación más rigurosa, se debería usar validación cruzada estratificada de k-folds o una división en tres conjuntos separados.

---

## Trabajo futuro

- Explorar el umbral de decisión óptimo usando la curva ROC
- Comparar contra otros clasificadores (Random Forest, SVM, regresión logística)
- Aplicar técnicas de balanceo de clases como SMOTE
- Implementar validación cruzada estratificada (5-fold) para métricas más robustas

---

## Tecnologías utilizadas

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f89b24?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.x-150458?logo=pandas&logoColor=white)

---

## Autor

**Manuel N. Palomino Villafane**  
Departamento de Ingeniería Electrónica — UTN FRBA  
mpalominovillafane@frba.utn.edu.ar
