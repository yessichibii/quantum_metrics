import numpy as np
from autograd import numpy as anp
from collections import defaultdict
import warnings
from typing import Optional, Union, Any, Tuple
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
import pennylane as qml

from ..base import QuantumBaseModel
from ..quantum.circuits import make_device, build_multiclase_qnode
from ..utils.params import init_params, cross_entropy
from ..utils.preprocesamiento import to_one_hot


class QMLMultiClase(QuantumBaseModel):
    """
    Quantum Machine Learning Multi-Class Classifier using parameterized quantum circuits.
    
    This classifier implements a quantum machine learning approach for multi-class
    classification tasks using PennyLane for quantum circuit simulation.
    
    Parameters
    ----------
    codigo : str, default="gray"
        Data encoding method for qubits. Options: "gray", "binario", "diag".
    noise : float, default=0.0
        Noise level simulated during circuit execution (0.0 to 1.0).
    backend : Any, optional
        Quantum backend or simulator. If None, uses package default backend.
    noise_model : Any, optional
        Noise model for the backend (if applicable).
    result : str, default="probs"
        Type of result requested: "probs" (probabilities) or "state" (state vector).
    shots : int, default=1024
        Number of executions (shots) when using sampling simulation.
    wr : float, default=0.5
        Weight range for weight optimizers.
    lr : float, default=0.1
        Learning rate for gradient-based optimizers.
    epochs : int, default=50
        Maximum number of training iterations.
    optimize : bool, default=True
        Whether to perform hyperparameter optimization during fit.
    random_state : int or None, default=None
        Random seed for reproducible results.
    k_folds : int, default=5
        Number of folds for cross-validation when using fit_validation.
    test_size : float, default=0.2
        Proportion used in hold-out validation (0.0 to 1.0).
    validation_split : float, default=0.2
        Proportion of training data to use for validation during fit.
    early_stopping : bool, default=False
        Whether to use early stopping during training.
    patience : int, default=10
        Number of epochs to wait before early stopping.
    verbose : bool, default=False
        Whether to print training progress.
    weights : array-like, optional
        Pesos específicos para el circuito cuántico. 
        Si es None, se inicializan aleatoriamente usando el rango definido por wr.
        Debe tener la longitud correcta según el número de características de los datos.
    use_weights : bool, default=True
        Si es True, utiliza optimización de pesos durante el entrenamiento.
        Si es False, solo ejecuta el circuito cuántico sin optimizar pesos.
    
    Attributes
    ----------
    params_ : ndarray or None
        Trained circuit parameters.
    best_params_ : dict or None
        Best parameters found through optimization/validation.
    history_ : list
        Training history (loss values per epoch).
    val_labels_ : ndarray or None
        Validation labels when using fit_validation.
    val_preds_ : ndarray or None
        Validation predictions when using fit_validation.
    n_features_in_ : int
        Number of features seen during fit.
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    
    Examples
    --------
    >>> from quantum_metric.models import QMLMultiClase
    >>> import numpy as np
    >>> X = np.random.random((100, 4))
    >>> y = np.random.randint(0, 3, 100)  # 3 classes
    >>> model = QMLMultiClase(codigo="gray", epochs=20)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    >>> probabilities = model.predict_proba(X)
    
    >>> # Using specific weights
    >>> initial_weights = np.random.random(4) * 0.1
    >>> model_with_weights = QMLMultiClase(codigo="gray", epochs=20, weights=initial_weights)
    >>> model_with_weights.fit(X, y)
    
    >>> # Using classifier without weight optimization
    >>> model_no_weights = QMLMultiClase(codigo="gray", use_weights=False)
    >>> model_no_weights.fit(X, y)
    """

    def __init__(self, codigo: str = "gray", noise: float = 0.0, backend: Optional[Any] = None,
                 noise_model: Optional[Any] = None, result: str = "probs", shots: int = 1024, wr: float = 0.5,
                 lr: float = 0.1, epochs: int = 50, optimize: bool = True, 
                 random_state: Optional[int] = None, k_folds: int = 5, 
                 test_size: float = 0.2, validation_split: float = 0.2,
                 early_stopping: bool = False, patience: int = 10, verbose: bool = False,
                 weights: Optional[Union[np.ndarray, list]] = None,
                 use_weights: bool = True):
        
        # Validate parameters
        self._validate_init_params(codigo, noise, result, shots, lr, epochs, 
                                 k_folds, test_size, validation_split, patience, weights, use_weights)
        
        super().__init__(lr=lr, epochs=epochs, optimize=optimize, random_state=random_state)
        
        self.codigo = codigo
        self.noise = noise
        self.backend = backend
        self.noise_model = noise_model
        self.result = result
        self.shots = shots
        self.wr = wr
        self.k_folds = k_folds
        self.test_size = test_size
        self.validation_split = validation_split
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        self.weights = weights
        self.use_weights = use_weights
        
        # Internal attributes
        self.folds = None
        self.train_data_ = None
        self.train_labels_ = None
        self.val_data_ = None
        self.val_labels_ = None
        self.best_params_ = None
        self.best_loss_ = None
        self.result_ = defaultdict(list)
        self.k_ = 1
        self._early_stopping_counter = 0

    def _validate_init_params(self, codigo: str, noise: float, result: str, 
                            shots: int, lr: float, epochs: int, k_folds: int,
                            test_size: float, validation_split: float, patience: int,
                            weights: Optional[Union[np.ndarray, list]],
                            use_weights: bool) -> None:
        """Validate initialization parameters."""
        
        # Validate codigo
        valid_codigos = ["gray", "binario", "diag"]
        if codigo not in valid_codigos:
            raise ValueError(
                f"codigo must be one of {valid_codigos}, got '{codigo}'"
            )
        
        # Validate noise
        if not isinstance(noise, (int, float)) or noise < 0.0 or noise > 1.0:
            raise ValueError(
                f"noise must be a float between 0.0 and 1.0, got {noise}"
            )
        
        # Validate result
        valid_results = ["probs", "state"]
        if result not in valid_results:
            raise ValueError(
                f"result must be one of {valid_results}, got '{result}'"
            )
        
        # Validate shots
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError(
                f"shots must be a positive integer, got {shots}"
            )
        
        # Validate learning rate
        if not isinstance(lr, (int, float)) or lr <= 0.0:
            raise ValueError(
                f"lr must be a positive number, got {lr}"
            )
        
        # Validate epochs
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(
                f"epochs must be a positive integer, got {epochs}"
            )
        
        # Validate k_folds
        if not isinstance(k_folds, int) or k_folds < 2:
            raise ValueError(
                f"k_folds must be an integer >= 2, got {k_folds}"
            )
        
        # Validate test_size
        if not isinstance(test_size, (int, float)) or test_size <= 0.0 or test_size >= 1.0:
            raise ValueError(
                f"test_size must be a float between 0.0 and 1.0, got {test_size}"
            )
        
        # Validate validation_split
        if not isinstance(validation_split, (int, float)) or validation_split <= 0.0 or validation_split >= 1.0:
            raise ValueError(
                f"validation_split must be a float between 0.0 and 1.0, got {validation_split}"
            )
        
        # Validate patience
        if not isinstance(patience, int) or patience <= 0:
            raise ValueError(
                f"patience must be a positive integer, got {patience}"
            )
        
        # Validate weights
        if weights is not None:
            if not isinstance(weights, (list, tuple, np.ndarray)):
                raise ValueError(
                    f"weights must be a list, tuple, or numpy array, "
                    f"got {type(weights)}"
                )
            
            # Convertir a numpy array para validación
            try:
                weights_array = np.array(weights)
            except Exception as e:
                raise ValueError(
                    f"weights must be convertible to numpy array: {e}"
                )
            
            # Validar que sean números finitos
            if not np.all(np.isfinite(weights_array)):
                raise ValueError(
                    "weights must contain only finite numbers"
                )
            
            # Validar que sean números reales
            if not np.issubdtype(weights_array.dtype, np.number):
                raise ValueError(
                    "weights must contain only numeric values"
                )
        
        # Validate use_weights
        if not isinstance(use_weights, bool):
            raise ValueError(
                f"use_weights must be a boolean, got {type(use_weights)}"
            )

    def _prepare_training_sets(self, data: np.ndarray, labels: np.ndarray, 
                             partitions: Optional[list] = None) -> None:
        """
        Prepare training sets for cross-validation.
        
        Parameters
        ----------
        data : ndarray
            Training data.
        labels : ndarray
            Training labels.
        partitions : list, optional
            Pre-defined partitions for cross-validation.
        """
        if partitions is not None:
            self.folds = partitions
            return
        
        if data is None or labels is None or len(data) == 0:
            raise ValueError("data and labels cannot be None or empty")
        
        if self.k_folds == 1:
            X_train, X_test, y_train, y_test = train_test_split(
                data, labels, test_size=self.test_size, 
                random_state=self.random_state, stratify=labels
            )
            self.folds = [(X_train, y_train, X_test, y_test)]
            return
        
        if self.k_folds > len(data):
            warnings.warn(
                f"k_folds={self.k_folds} > samples={len(data)}; "
                f"adjusting to Leave-One-Out CV.", UserWarning
            )
            self.k_folds = len(data)

        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        self.folds = [(data[train], labels[train], data[test], labels[test]) 
                     for train, test in kf.split(data)]

    def fit_validation(self, X: np.ndarray, y: np.ndarray, 
                      partitions: Optional[list] = None) -> 'QMLMultiClase':
        """
        Fit the model with internal validation (hold-out, k-fold or LOOCV).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (class labels).
        partitions : list, optional
            Pre-defined partitions for cross-validation.
            
        Returns
        -------
        self : QMLMultiClase
            Returns self for method chaining.
        """
        # Validate input
        X, y = self._validate_input(X, y, reset=True)
        
        if len(X) == 0:
            raise ValueError("X cannot be empty")
        
        # Split data for validation
        self.train_data_, self.val_data_, self.train_labels_, self.val_labels_ = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        self._prepare_training_sets(self.train_data_, self.train_labels_, partitions=partitions)
        
        if not self.folds:
            raise ValueError("No folds available for training")
        
        # Train on each fold
        for i, (X_train, y_train, X_test, y_test) in enumerate(self.folds):
            if self.verbose:
                print(f"\n🌀 Fold {i+1}/{len(self.folds)}")
            
            try:
                self.fit(X_train, y_train, X_test, y_test)
            except Exception as e:
                warnings.warn(
                    f"fit_validation: failed on fold {i+1} -> {e}. "
                    f"Continuing with next folds.", UserWarning
                )
                continue
        
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, 
           y_val: Optional[np.ndarray] = None) -> 'QMLMultiClase':
        """
        Fit the quantum model to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (class labels).
        X_val : array-like of shape (n_val_samples, n_features), optional
            Validation data.
        y_val : array-like of shape (n_val_samples,), optional
            Validation target values.
            
        Returns
        -------
        self : QMLMultiClase
            Returns self for method chaining.
        """
        # Validate input
        X, y = self._validate_input(X, y, reset=True)
        
        if len(X) == 0:
            raise ValueError("X cannot be empty")
        
        # Validate multi-class classification
        if self.n_classes_ < 2:
            raise ValueError(
                f"QMLMultiClase requires at least 2 classes, "
                f"but found {self.n_classes_} classes: {self.classes_}"
            )
        
        # Prepare validation data if not provided
        if X_val is None or y_val is None:
            if self.validation_split > 0:
                X, X_val, y, y_val = train_test_split(
                    X, y, test_size=self.validation_split, 
                    random_state=self.random_state, stratify=y
                )
        
        # Validate validation data if provided
        if X_val is not None and y_val is not None:
            X_val, y_val = check_X_y(X_val, y_val, accept_sparse=False, dtype=np.float64)
            check_classification_targets(y_val)
        
        try:
            m, qubits_dato = X.shape
        except Exception as e:
            raise ValueError(
                f"X must be a 2D array with shape (n_samples, n_features). "
                f"Error: {e}"
            )

        # Calculate quantum circuit parameters
        qubits_qram = int(np.ceil(np.log2(m)))
        qubits_label = int(np.ceil(np.log2(self.n_classes_)))
        n_totales = qubits_qram + qubits_dato + qubits_label + 1
        
        # Debug information
        if self.verbose:
            print(f"Circuit parameters: m={m}, qubits_dato={qubits_dato}, qubits_qram={qubits_qram}, qubits_label={qubits_label}, n_totales={n_totales}")
        
        if n_totales > 20:  # Reasonable limit for simulation
            warnings.warn(
                f"Large quantum circuit with {n_totales} qubits. "
                f"Simulation may be slow.", UserWarning
            )

        # Initialize quantum device and parameters
        dev = make_device(n_totales, backend=self.backend, shots=self.shots, noise_model=self.noise_model)
        params = init_params(qubits_dato, init=self.weights, rng=self.wr)
        
        # Initialize optimizer only if needed
        opt = None
        if self.use_weights:
            opt = qml.AdamOptimizer(stepsize=self.lr)
        
        # Build quantum circuit
        try:
            circuit = build_multiclase_qnode(
                X, y, n_totales, qubits_qram, qubits_dato, qubits_label, dev, 
                codigo=self.codigo, noise=self.noise, result=self.result
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build QNode: {e}")

        # Handle circuit wrapping if needed
        circuit_callable = self._wrap_circuit(circuit)

        # Define cost function
        def cost(p, tests, y_test):
            return self._compute_cost(p, tests, y_test, circuit_callable)

        # Store training data for prediction
        self.train_data_ = X
        self.train_labels_ = y
        
        # Training loop
        loss_history = []
        param_history = []
        best_loss = None
        best_params = None
        self._early_stopping_counter = 0

        if self.use_weights:
            # Training with optimizer
            for epoch in range(self.epochs):
                try:
                    params, loss_new = opt.step_and_cost(lambda v: cost(v, X_val if X_val is not None else X, 
                                                                      y_val if y_val is not None else y), params)
                except Exception as e:
                    warnings.warn(
                        f"Optimizer failed at epoch {epoch}: {e}. Stopping training.", 
                        UserWarning
                    )
                    break
                
                loss_history.append(loss_new)
                param_history.append(params.copy() if hasattr(params, "copy") else params)

                # Track best parameters
                if best_loss is None or (loss_new is not None and loss_new < best_loss):
                    best_loss = loss_new
                    best_params = params
                    self._early_stopping_counter = 0
                else:
                    self._early_stopping_counter += 1

                # Early stopping
                if self.early_stopping and self._early_stopping_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

                if self.verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss = {loss_new:.6f}")
        else:
            # No optimization - just evaluate with initial weights
            if self.verbose:
                print("Running classifier without weight optimization...")
            
            try:
                loss_new = cost(params, X_val if X_val is not None else X, 
                              y_val if y_val is not None else y)
                loss_history.append(loss_new)
                param_history.append(params.copy() if hasattr(params, "copy") else params)
                best_loss = loss_new
                best_params = params
                
                if self.verbose:
                    print(f"Initial evaluation - Loss = {loss_new:.6f}")
                    
            except Exception as e:
                warnings.warn(
                    f"Failed to evaluate circuit: {e}. Using default parameters.", 
                    UserWarning
                )
                loss_history.append(float('inf'))
                param_history.append(params)
                best_loss = float('inf')
                best_params = params

        # Store results
        self.params_ = best_params
        self.history_ = loss_history
        
        if self.best_loss_ is None or (best_loss is not None and best_loss < self.best_loss_):
            self.best_loss_ = best_loss
            self.best_params_ = best_params

        self.result_[self.k_] = {
            'params_': param_history,
            'history_': loss_history,
            'best_params_': best_params,
            'best_loss_': best_loss
        }
        self.k_ += 1
        
        return self

    def _wrap_circuit(self, circuit):
        """Wrap circuit if it's a list or not callable."""
        if isinstance(circuit, list) or not callable(circuit):
            def _wrapped_circuit(test, params):
                try:
                    outs = []
                    for q in circuit:
                        if callable(q):
                            outs.append(q(test=test, params=params))
                        else:
                            outs.append(q)
                    return np.asarray(outs)
                except Exception as e:
                    raise RuntimeError(f"Error in wrapped circuit: {e}")
            return _wrapped_circuit
        else:
            return circuit

    def _compute_cost(self, p, tests, y_test, circuit_callable):
        """Compute cost function for training."""
        probs = []
        for test in tests:
            try:
                predict = circuit_callable(test=test, params=p)
                probs.append(np.asarray(predict))
            except Exception as e:
                warnings.warn(
                    f"Failed to execute QNode for test sample: {e}. "
                    f"Using uniform probability.", UserWarning
                )
                n_classes = self.n_classes_
                probs.append(np.ones(n_classes) / n_classes)
        
        try:
            probs_arr = np.vstack(probs)
        except Exception:
            probs_arr = np.array(probs)
        
        try:
            return cross_entropy(to_one_hot(y_test), probs_arr)
        except Exception as e:
            warnings.warn(f"Failed in cross_entropy: {e}. Returning high loss.", UserWarning)
            return np.array(1e6)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        if len(X) == 0:
            return np.array([])
        
        train_data = self.train_data_ if self.train_data_ is not None else None
        train_labels = self.train_labels_ if self.train_labels_ is not None else None
        
        if train_data is None or train_labels is None:
            raise ValueError(
                "Model must be fitted with training data before prediction. "
                "Use fit() or fit_validation() first."
            )
        
        try:
            m, qubits_dato = train_data.shape
        except Exception:
            raise ValueError("Training data must be a 2D array")
        
        qubits_qram = int(np.ceil(np.log2(m)))
        qubits_label = int(np.ceil(np.log2(self.n_classes_)))
        n_totales = qubits_qram + qubits_dato + qubits_label + 1
        
        # Debug information
        if self.verbose:
            print(f"Prediction circuit parameters: m={m}, qubits_dato={qubits_dato}, qubits_qram={qubits_qram}, qubits_label={qubits_label}, n_totales={n_totales}")
        
        dev = make_device(n_totales, backend=self.backend, shots=self.shots, 
                         noise_model=self.noise_model)
        
        try:
            circuit = build_multiclase_qnode(
                train_data, train_labels, n_totales, qubits_qram, qubits_dato, qubits_label, dev, 
                codigo=self.codigo, noise=self.noise, result=self.result
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build QNode for prediction: {e}")

        circuit_callable = self._wrap_circuit(circuit)

        if self.verbose:
            print(f"circuit_callable: {circuit_callable}")

        params = self.best_params_
        if params is None:
            try:
                arr = np.array([self.result_[r]['best_params_'] for r in self.result_ 
                               if 'best_params_' in self.result_[r]])
                if arr.size > 0:
                    params = np.mean(arr, axis=0)
                    warnings.warn(
                        "best_params_ not found, using average of fold parameters.", 
                        UserWarning
                    )
                else:
                    warnings.warn(
                        "No trained parameters found; returning default predictions (class 0).", 
                        UserWarning
                    )
                    return np.zeros(len(X), dtype=int)
            except Exception:
                warnings.warn(
                    "Error getting parameter average; returning zeros.", 
                    UserWarning
                )
                return np.zeros(len(X), dtype=int)

        # Make predictions
        preds = []
        
        for i in X:
            try:
                pred = circuit_callable(test=i, params=params)
                pred = np.asarray(pred)
                
                # Handle different output shapes
                if pred.ndim == 2 and pred.shape[0] > 1:
                    pred = pred[0]
                
                preds.append(int(np.argmax(pred)))
            except Exception as e:
                warnings.warn(
                    f"Prediction failed for sample: {e}. Predicting class 0.", 
                    UserWarning
                )
                preds.append(0)

        return np.array(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict probabilities for.
            
        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Class probabilities for each sample.
        """
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        
        if len(X) == 0:
            return np.array([]).reshape(0, self.n_classes_)
        
        # Use training data from fit_validation if available
        train_data = self.train_data_ if self.train_data_ is not None else None
        train_labels = self.train_labels_ if self.train_labels_ is not None else None
        
        if train_data is None or train_labels is None:
            raise ValueError(
                "Model must be fitted with training data before prediction. "
                "Use fit() or fit_validation() first."
            )
        
        try:
            m, qubits_dato = train_data.shape
        except Exception:
            raise ValueError("Training data must be a 2D array")
        
        qubits_qram = int(np.ceil(np.log2(m)))
        qubits_label = int(np.ceil(np.log2(self.n_classes_)))
        n_totales = qubits_qram + qubits_dato + qubits_label + 1
        
        # Debug information
        if self.verbose:
            print(f"Probability prediction circuit parameters: m={m}, qubits_dato={qubits_dato}, qubits_qram={qubits_qram}, qubits_label={qubits_label}, n_totales={n_totales}")
        
        dev = make_device(n_totales, backend=self.backend, shots=self.shots, 
                         noise_model=self.noise_model)
        
        try:
            circuit = build_multiclase_qnode(
                train_data, train_labels, n_totales, qubits_qram, qubits_dato, qubits_label, dev, 
                codigo=self.codigo, noise=self.noise, result=self.result
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build QNode for probability prediction: {e}")

        circuit_callable = self._wrap_circuit(circuit)
        
        # Get parameters for prediction
        params = self.best_params_
        if params is None:
            try:
                arr = np.array([self.result_[r]['best_params_'] for r in self.result_ 
                               if 'best_params_' in self.result_[r]])
                if arr.size > 0:
                    params = np.mean(arr, axis=0)
                else:
                    warnings.warn(
                        "No trained parameters found; returning uniform probabilities.", 
                        UserWarning
                    )
                    return np.ones((len(X), self.n_classes_)) / self.n_classes_
            except Exception:
                warnings.warn(
                    "Error getting parameters; returning uniform probabilities.", 
                    UserWarning
                )
                return np.ones((len(X), self.n_classes_)) / self.n_classes_

        # Make probability predictions
        probas = []
        for i in X:
            try:
                pred = circuit_callable(test=i, params=params)
                pred = np.asarray(pred)
                
                # Handle different output shapes
                if pred.ndim == 2 and pred.shape[0] > 1:
                    pred = pred[0]
                
                # Ensure probabilities sum to 1
                pred = pred / np.sum(pred) if np.sum(pred) > 0 else pred
                probas.append(pred)
            except Exception as e:
                warnings.warn(
                    f"Probability prediction failed for sample: {e}. "
                    f"Using uniform probabilities.", UserWarning
                )
                probas.append(np.ones(self.n_classes_) / self.n_classes_)

        return np.array(probas)
