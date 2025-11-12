# 🚀 Quantum Machine Learning Library - Architecture Guide

### **1. Clase Base Compatible con scikit-learn**
- ✅ Hereda de `BaseEstimator` y `ClassifierMixin`
- ✅ Implementa métodos estándar: `fit()`, `predict()`, `predict_proba()`, `score()`
- ✅ Atributos con convención `_` (terminan con underscore)
- ✅ Validación robusta de entrada usando `check_X_y` y `check_array`

### **2. Validaciones**
- ✅ Validación completa de parámetros de inicialización
- ✅ Validación de tipos de datos y rangos
- ✅ Validación de formas de datos
- ✅ Validación específica para clasificación binaria
- ✅ Manejo apropiado de errores con mensajes informativos

### **3. Manejo de Errores**
- ✅ Uso de `ValueError` para errores de parámetros
- ✅ Uso de `warnings` para situaciones recuperables
- ✅ Mensajes de error descriptivos
- ✅ Validación de estado del modelo antes de predicciones


### **Funcionalidades básicas:**
- ✅ `predict()`: Predicciones de clase
- ✅ `predict_proba()`: Predicciones de probabilidad
- ✅ Early stopping para evitar sobreajuste
- ✅ Validación de tamaño de circuito cuántico
- ✅ Soporte para random_state para reproducibilidad
- ✅ Modo verbose para seguimiento del entrenamiento
- ✅ **Pesos personalizados**: Soporte para pesos específicos del circuito cuántico
- ✅ **Control de optimización**: Opción para usar solo el clasificador sin optimizar pesos

## 📚 **Estructura de Archivos**

```
src/quantum_information/
├── __init__.py                 # Punto de entrada del nuevo paquete
├── circuits/                   # Construcción genérica de circuitos
│   ├── __init__.py
│   └── circuits.py
├── encoding/                   # Codificación de datos a estados cuánticos
│   ├── __init__.py
│   ├── qram.py
│   ├── advanced/
│   │   ├── __init__.py
│   │   ├── BasisEncoding.py
│   │   ├── DataEncoding.py
│   │   └── QRAM.py
│   └── deprecated/             # Implementaciones heredadas (comentadas como deprecated)
│       ├── __init__.py
│       ├── data_encoding.py
│       ├── performance.py
│       ├── QFT_arithmetic.py
│       └── visualization.py
├── metrics/                    # Métricas y modelos basados en distancias
│   ├── __init__.py
│   ├── base.py
│   ├── distances.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── parametricos.py
│   │   ├── qml.py
│   │   ├── qml_biclase.py
│   │   └── qml_multiclase.py
│   └── validation/
│       ├── __init__.py
│       └── cross_validation.py
├── preprocessing/              # Pipelines clásicos antes de codificar
│   ├── __init__.py
│   └── data.py
└── utils/                      # Utilidades transversales
    ├── __init__.py
    ├── params.py
    └── utils.py
```

## 🚀 **Recomendaciones Adicionales**

### **1. Testing**
```bash
# Instalar dependencias de desarrollo
pip install -e ".[dev]"

# Ejecutar tests
pytest tests/ -v --cov=quantum_information
```

### **2. Documentación**
- ✅ Docstrings completos con formato NumPy
- ✅ Ejemplos de uso en `examples/`
- ✅ Configuración de Sphinx para documentación

### **3. CI/CD**
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run tests
        run: pytest tests/ -v --cov=quantum_information
```

## 🔍 **Validaciones Implementadas**

### **Validaciones de Parámetros:**
- ✅ Tipos de datos correctos
- ✅ Rangos válidos para parámetros numéricos
- ✅ Valores permitidos para parámetros categóricos
- ✅ Consistencia entre parámetros relacionados

```python
# Validaciones automáticas en todos los métodos
model = QMLBiClase(noise=1.5)  # ❌ ValueError: noise must be between 0.0 and 1.0
model.fit(X, y)                # ❌ ValueError: X must be 2D array
model.predict(X_new)           # ❌ ValueError: Model not fitted yet
```

### **Validaciones de Datos:**
- ✅ Formas de arrays correctas
- ✅ Tipos de datos apropiados
- ✅ Número mínimo de muestras y características
- ✅ Validación específica para clasificación binaria

### **Validaciones de Estado:**
- ✅ Modelo debe estar entrenado antes de predicciones
- ✅ Datos de entrenamiento disponibles para predicciones
- ✅ Parámetros entrenados válidos

### **Pesos Personalizados:**
- ✅ Soporte para `weights` en todos los modelos cuánticos
- ✅ Validación automática de longitud y tipo de pesos
- ✅ Fallback a inicialización aleatoria si no se proporcionan
- ✅ Validación de números finitos y válidos

### **Control de Optimización:**
- ✅ Parámetro `use_weights` para controlar el entrenamiento
- ✅ Modo sin optimización para evaluar clasificador base
- ✅ Evaluación única con pesos iniciales cuando `use_weights=False`
- ✅ Compatibilidad completa con la interfaz estándar

```python
# Uso de pesos personalizados
import numpy as np
from quantum_metric.models import QMLBiClase

# Pesos específicos
weights = np.array([0.1, -0.2, 0.3, -0.1])
model = QMLBiClase(
    codigo="gray", 
    epochs=20, 
    weights=weights
)

# Clasificador sin optimización de pesos
model_no_weights = QMLBiClase(
    codigo="gray", 
    use_weights=False
)

# Validación automática
try:
    model = QMLBiClase(weights=[1, 2, 3])  # ❌ Longitud incorrecta
except ValueError as e:
    print(e)  # "Los pesos deben tener longitud X, pero se proporcionaron 3 pesos"

# Uso del modelo multiclase
from quantum_metric.models import QMLMultiClase

# Clasificación multiclase
X_multi, y_multi = make_classification(n_samples=100, n_features=4, n_classes=3)
model_multi = QMLMultiClase(codigo="gray", epochs=20)
model_multi.fit(X_multi, y_multi)
predictions_multi = model_multi.predict(X_multi)
probabilities_multi = model_multi.predict_proba(X_multi)
```

```python
# Mensajes de error claros y útiles
try:
    model = QMLBiClase(codigo="invalid")
except ValueError as e:
    print(e)  # "codigo must be one of ['gray', 'binario', 'diag'], got 'invalid'"
```
