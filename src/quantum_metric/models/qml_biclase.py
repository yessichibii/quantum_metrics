import numpy as np
from autograd import numpy as anp
from collections import defaultdict
import warnings
from sklearn.model_selection import KFold, train_test_split
import pennylane as qml
from ..base import QuantumBaseModel
from ..quantum.circuits import make_device, build_biclase_qnode
from ..utils.params import init_params, cross_entropy
from ..utils.preprocesamiento import to_one_hot

class QMLBiClase(QuantumBaseModel):

    def __init__(self, codigo="gray", noise=0.0, backend=None,
                 noise_model=None, result="probs", shots=1024,
                 lr=0.1, epochs=50, optimize=True, rng=0.5, k_folds=5, test_size=0.2):
        super().__init__(lr, epochs, optimize, k_folds, test_size)
        self.codigo = codigo
        self.noise = noise
        self.backend = backend
        self.noise_model = noise_model
        self.result = result
        self.shots = shots
        self.rng = rng
        self.k_folds = k_folds
        self.test_size = test_size

        self.folds = None
        self.train_data_ = None        # array con los ejemplos de entrenamiento (X_train)
        self.train_labels_ = None      # etiquetas (one-hot) de training
        self.val_data_ = None      # etiquetas (one-hot) de training
        self.val_labels_ = None      # etiquetas (one-hot) de training

        self.best_params_ = None
        self.best_lost_ = None
        self.result_ = defaultdict(list)
        self.k_ = 1

    def _prepare_training_sets(self,  data, labels, partitions=None):
        """Prepara train_data_, train_labels_ (one-hot) y el map clase->indices."""
        if partitions is not None:
            self.folds = partitions
            return
        
        if data is None or labels is None or len(data) == 0:
            warnings.warn("_prepare_training_sets: data/labels vacíos.", UserWarning)
            self.folds = []
            return
        
        if self.k_folds == 1:
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=self.test_size, random_state=42)
            self.folds = [(X_train, y_train, X_test, y_test)]
            return
        
        if self.k_folds > len(data):
            warnings.warn(f"k_folds={self.k_folds} > muestras={len(data)}; se ajusta a LOOCV.", UserWarning)
            self.k_folds = len(data)

        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        self.folds = [(data[train], labels[train], data[test], labels[test]) for train, test in kf.split(data)]

    def fit_validation(self, data, labels, partitions=None):

        if data is None or labels is None:
            raise ValueError("fit_validation: 'data' y 'labels' no pueden ser None.")

        self.train_data_, self.val_data_, self.train_labels_, self.val_labels_ = train_test_split(data, labels, test_size=self.test_size, random_state=42)
        self._prepare_training_sets(self.train_data_, self.train_labels_, partitions=partitions)
        
        if not self.folds:
            warnings.warn("fit_validation: no hay folds para entrenar.", UserWarning)
            return self

        for i, (X_train, y_train, X_test, y_test) in enumerate(self.folds):
            print(f"\n🌀 Fold {i+1}/{len(self.folds)}")
            try:
                self.fit(X_train, y_train, X_test, y_test)
            except Exception as e:
                warnings.warn(f"fit_validation: fallo en fold {i+1} -> {e}. Se continúa con siguientes folds.", UserWarning)
                continue
        return self
    
    def fit(self, X_train, y_train, X_test, y_test):

        if X_train is None or y_train is None:
            raise ValueError("fit: X_train y y_train son requeridos.")
        if len(X_train) == 0:
            raise ValueError("fit: X_train vacío.")
        try:
            m, qubits_dato = X_train.shape
        except Exception:
            raise ValueError("fit: X_train debe ser array 2D con shape (n_samples, n_features).")

        qubits_qram = int(np.ceil(np.log2(m)))
        n_totales = qubits_qram + qubits_dato + 2

        dev = make_device(n_totales, backend=self.backend, shots=self.shots, noise_model=self.noise_model)
        params = init_params(qubits_dato, rng=self.rng)
        opt = qml.AdamOptimizer(stepsize=self.lr)


        try:
            circuit = build_biclase_qnode(X_train, y_train, n_totales, qubits_qram, qubits_dato, dev, codigo=self.codigo, noise=self.noise, result=self.result)
        except Exception as e:
            raise RuntimeError(f"fit: fallo al construir QNode -> {e}")
        

        if isinstance(circuit, list) or not callable(circuit):
            # intentar envolver lista de QNodes o lista de arrays
            def _wrapped_circuit(test, params):
                # si circuit es lista de QNodes, llamarlos; si son arrays devueltos, concatenarlos/stack
                try:
                    outs = []
                    for q in circuit:
                        if callable(q):
                            outs.append(q(test=test, params=params))
                        else:
                            outs.append(q)
                    # convertir a array (n_tests, n_classes) o (n_classes,)
                    return np.asarray(outs)
                except Exception as e:
                    raise RuntimeError(f"_wrapped_circuit: error llamando circuito envuelto -> {e}")
            circuit_callable = _wrapped_circuit
        else:
            circuit_callable = circuit


        def cost(p, tests, y_test):
            probs = []
            for test in tests:
                try:
                    predict = circuit_callable(test=test, params=p)
                    probs.append(np.asarray(predict))
                except Exception as e:
                    warnings.warn(f"cost: fallo al ejecutar QNode para un test -> {e}. Se usa prob uniforme.", UserWarning)
                    # fallback: prob uniforme según número de clases deducidas de y_test one-hot
                    try:
                        n_classes = to_one_hot(y_test).shape[1]
                    except Exception:
                        n_classes = 2
                    probs.append(np.ones(n_classes) / n_classes)
                        # asegurar forma (n_samples, n_classes)
            try:
                probs_arr = np.vstack(probs)
            except Exception:
                probs_arr = np.array(probs)
            # cross_entropy espera one-hot y probs shape compatibles
            try:
                return cross_entropy(to_one_hot(y_test), probs_arr)
            except Exception as e:
                warnings.warn(f"cost: fallo en cross_entropy -> {e}. Se retorna pérdida alta.", UserWarning)
                return np.array(1e6)   
            # return cross_entropy(to_one_hot(y_test), probs)

        loss_cost = []
        list_params = []
        best_loss = None
        best_param = None

        for epoch in range(self.epochs):
            try:
                params, loss_new = opt.step_and_cost(lambda v: cost(v, X_test, y_test), params)
            except Exception as e:
                warnings.warn(f"fit: optimizador falló en epoch {epoch} -> {e}. Se detiene iteración.", UserWarning)
                break
            
            loss_cost.append(loss_new)
            list_params.append(params.copy() if hasattr(params, "copy") else params)

            if best_loss is None or (loss_new is not None and loss_new < best_loss):
                best_loss = loss_new
                best_param = params

        try:
            if self.best_lost_ is None or (best_loss is not None and best_loss < self.best_lost_):
                self.best_lost_ = best_loss
                self.best_params_ = best_param
        except Exception:
            warnings.warn("fit: no se pudieron actualizar best_params_/best_lost_.", UserWarning)


        self.result_[self.k_] = {
            'params_': list_params,
            'history_': loss_cost,
            'best_params_': best_param,
            'best_loss_': best_loss
        }
        self.k_ += 1
        return self

    def predict(self, test = None, train = None, y_train = None, method = "max"):
        if test is None:
            test = self.val_data_

        if train is None:
            train = self.train_data_

        if y_train is None:
            y_train = self.train_labels_

        if train is None or y_train is None:
            raise ValueError("predict: se requiere 'train' y 'y_train' (o haber ejecutado fit_validation).")

        try:
            m, qubits_dato = train.shape
        except Exception:
            raise ValueError("predict: 'train' debe ser array 2D.")
        
        qubits_qram = int(np.ceil(np.log2(m)))
        n_totales = qubits_qram + qubits_dato + 2
        dev = make_device(n_totales, backend=self.backend, shots=self.shots, noise_model=self.noise_model)
        
        try:
            circuit = build_biclase_qnode(train, y_train, n_totales, qubits_qram, qubits_dato, dev, codigo=self.codigo, noise=self.noise, result=self.result)
        except Exception as e:
            raise RuntimeError(f"predict: fallo al construir QNode -> {e}")

        if isinstance(circuit, list) or not callable(circuit):
            def _wrapped_circuit(test, params):
                outs = []
                for q in circuit:
                    try:
                        outs.append(q(test=test, params=params) if callable(q) else q)
                    except Exception:
                        outs.append(None)
                return np.asarray(outs)
            circuit_callable = _wrapped_circuit
        else:
            circuit_callable = circuit

        params = self.best_params_
        if params is None:
            try:
                arr = np.array([self.result_[r]['best_params_'] for r in self.result_ if 'best_params_' in self.result_[r]])
                if arr.size > 0:
                    params = np.mean(arr, axis=0)
                    warnings.warn("predict: best_params_ no existe, usando promedio de best_params_ por fold.", UserWarning)
                else:
                    warnings.warn("predict: no hay parámetros entrenados; retornando predicciones por defecto (ceros).", UserWarning)
                    return [0 for _ in range(len(test))]
            except Exception:
                warnings.warn("predict: error al obtener promedio de parámetros; retornando ceros.", UserWarning)
                return [0 for _ in range(len(test))]


        if method == "mean":
            try:
                arr = np.array([self.result_[r]['best_params_'] for r in self.result_ if 'best_params_' in self.result_[r]])
                if arr.size > 0:
                    params = np.mean(arr, axis=0)
            except Exception:
                warnings.warn("predict: no se pudo calcular mean params; se usa params seleccionados.", UserWarning)


        preds = []
        for i in test:
            try:
                pred = circuit_callable(test = i, params = params)
                pred = np.asarray(pred)
                # si devuelve vector de probabilidades, escoger argmax; si devuelve matriz, usar primera fila
                if pred.ndim == 2 and pred.shape[0] > 1:
                    pred = pred[0]
                preds.append(int(np.argmax(pred)))
            except Exception as e:
                warnings.warn(f"predict: fallo en inferencia para un sample -> {e}. Se predice clase 0.", UserWarning)
                preds.append(0)

        return preds

