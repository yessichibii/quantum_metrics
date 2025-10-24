# Quantum Metrics

**Quantum Metrics** es una librería en Python para calcular métricas cuánticas de distancia  utilizando QRAM con PennyLane, se puede emplear un esquema hibrido obteniendo el resultado de la distancia o agregarlo como una subrutina.

# 🚀 Instalación de la biblioteca
La forma más sencilla es instalar desde PyPI:

```bash
pip install quantum-metrics
```

# 📦 Dependencias

La librería depende de:
Python >= 3.8
NumPy
SciPy
PennyLane
Qiskit
qiskit_aer


Se instalarán automáticamente con pip install -e ., pero también puedes instalarlas manualmente:

pip install numpy scipy pennylane

# ⚡ Uso básico

Puedes importar las funciones directamente desde el paquete:



## 🎯 Modelos de Machine Learning Cuántico

La librería incluye modelos de clasificación cuántica con soporte para parámetros iniciales personalizados:

```python
from quantum_metric.models import QMLBiClase
import numpy as np

# Datos de ejemplo
X = np.random.random((100, 4))
y = np.random.randint(0, 2, 100)

# Modelo con inicialización aleatoria (por defecto)
model = QMLBiClase(codigo="gray", epochs=20)
model.fit(X, y)

# Modelo con pesos específicos
initial_weights = np.random.random(4) * 0.1
model_with_weights = QMLBiClase(
    codigo="gray", 
    epochs=20, 
    weights=initial_weights
)
model_with_weights.fit(X, y)

# Modelo sin optimización de pesos (solo clasificador)
model_no_weights = QMLBiClase(
    codigo="gray", 
    use_weights=False
)
model_no_weights.fit(X, y)

# Predicciones
predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

### 🔧 Pesos y Optimización

Los modelos cuánticos soportan pesos personalizados y control de optimización:

- **`weights`**: Array de pesos específicos para el circuito cuántico
- Si es `None` (por defecto), se inicializan aleatoriamente
- Debe tener la longitud correcta según el número de características
- Se valida automáticamente que sean números finitos y válidos

- **`use_weights`**: Controla si se optimizan los pesos durante el entrenamiento
- Si es `True` (por defecto), utiliza optimización de pesos
- Si es `False`, solo ejecuta el circuito cuántico sin optimizar pesos
- Útil para evaluar el rendimiento del clasificador base sin entrenamiento

# 📂 Estructura del proyecto
quantum-metrics/
│
├── src/
│   └── quantum_metrics/
│       ├── __init__.py
│       ├── distances.py
│       ├── initialize.py
│       └── utils.py
│
├── examples/
│   └── demo.py
│   └── qml/
│       ├── modelo_biclase.py
│       ├── modelo_asociativo.py
│       └── modelo_multilabel.py
│
├── tests/
│   └── test_distances.py
│
├── pyproject.toml
├── README.md
├── LICENSE

# 🧪 Pruebas

Ejecuta los tests con pytest:

pytest

# 📬 Contribuciones

¡Las contribuciones son bienvenidas!

https://github.com/yessichibii/quantum_metrics

Por favor abre un issue o un pull request en GitHub


