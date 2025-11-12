# QMLBiClase — Clasificador binario cuántico

Resumen
-------
QMLBiClase es un clasificador binario basado en métricas que emplea la métrica MBA que incluye un proceso de oprimización de pesos para los rasgos implementado en PennyLane.

Provee una API compatible con scikit-learn: fit / predict además de un método de validación interna fit_validation y hooks para optimización de hiperparámetros.

Estructura y objetivos
- Construye un QNode (o varios) que codifican los datos y generan probabilidades por clase.
- Usa optimizadores clásicos (p. ej. Adam) para ajustar parámetros del circuito.
- Soporta validación interna (k-fold, hold-out o LOOCV) y se integra con fit_validation.

Instalación / requisitos
- Python 3.8+
- PennyLane (pennylane)
- autograd, numpy, scikit-learn
- Repositorio con estructura:
  - src/quantum_metric/models/qml_biclase.py  (implementación)


API principal (resumen)
-----------------------
Constructor:
- QMLBiClase(codigo="gray", noise=0.0, backend=None, noise_model=None,
             result="probs", shots=1024, lr=0.1, epochs=50,
             optimize=True, rng=0.5, k_folds=5, test_size=0.2)

Atributos relevantes:
- best_params_: parámetros del mejor modelo tras validación
- best_lost_: pérdida asociada a best_params_
- result_: dict con histórico por fold
- train_data_, val_data_, train_labels_, val_labels_

Métodos:
- fit(X_train, y_train, X_test, y_test)
  Entrena usando las particiones provistas. Actualiza result_ y best_params_.
- fit_validation(data, labels, partitions=None)
  Realiza particionado (hold-out o k-fold) y entrena por fold.
- predict(test=None, train=None, y_train=None, method="max")
  Devuelve etiquetas predichas (0/1) usando los params entrenados. Requiere train+y_train
  (o haber ejecutado fit_validation antes).

Uso recomendado (ejemplo breve)
------------------------------
1) Validación externa usando cross_validate_biclase:
```python
from quantum_metric.models.qml_biclase import QMLBiClase
from quantum_metric.validation.cross_validation import cross_validate_biclase

model = QMLBiClase(lr=0.1, epochs=10, optimize=True)
res = cross_validate_biclase(model, X, y, k_folds=3, epochs=5, lr=0.1, rng=0.5, optimize=True)
print(res["mean_loss"], res["best_params"])
```

2) Validación interna con un objeto:
```python
model = QMLBiClase(epochs=10, lr=0.1)
model.fit_validation(X, y, k_folds=3)
preds = model.predict(test=X_val, train=model.train_data_, y_train=model.train_labels_)
```

Notas prácticas y recomendaciones
--------------------------------
- Para pruebas rápidas use pocas epochs (2-10) y un backend simulador (ej. default.qubit).
- Para reproducibilidad, pasar rng como entero (semilla).
- Estandarizar X (StandardScaler) antes de codificar.
- Si experimentas errores en construcción de QNode, revisa shapes y tipo de salida esperada (1D p(y=1) o 2D (n,2)).

Depuración y troubleshooting
---------------------------
- Si predict devuelve ceros o warnings sobre best_params_, revisa que fit/fit_validation se hayan ejecutado correctamente y que no haya fallos en la construcción del QNode.
- Si el optimizador falla en alguna epoch, captura warnings y reduce lr o epochs.
- Si tu entorno no tiene PennyLane o backend compatible, instala o usa un simulador (pennylane default.qubit).

Más documentación
-----------------
- Revisar docstrings en src/quantum_metric/models/qml_biclase.py
- Ejemplos ejecutables en examples/qml/biclase_demo.ipynb (notebook generado).