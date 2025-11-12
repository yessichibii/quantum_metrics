"""
Common validation utilities for quantum machine learning models.
"""

import numpy as np
from typing import Any, Union, Optional
import warnings


def validate_quantum_parameters(**kwargs) -> None:
    """
    Validate quantum-specific parameters.
    
    Parameters
    ----------
    **kwargs : dict
        Parameters to validate.
        
    Raises
    ------
    ValueError
        If any parameter is invalid.
    """
    
    # Validate codigo parameter
    if 'codigo' in kwargs:
        codigo = kwargs['codigo']
        valid_codigos = ["gray", "binario", "diag"]
        if codigo not in valid_codigos:
            raise ValueError(
                f"codigo must be one of {valid_codigos}, got '{codigo}'"
            )
    
    # Validate noise parameter
    if 'noise' in kwargs:
        noise = kwargs['noise']
        if not isinstance(noise, (int, float)) or noise < 0.0 or noise > 1.0:
            raise ValueError(
                f"noise must be a float between 0.0 and 1.0, got {noise}"
            )
    
    # Validate shots parameter
    if 'shots' in kwargs:
        shots = kwargs['shots']
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError(
                f"shots must be a positive integer, got {shots}"
            )
    
    # Validate result parameter
    if 'result' in kwargs:
        result = kwargs['result']
        valid_results = ["probs", "state"]
        if result not in valid_results:
            raise ValueError(
                f"result must be one of {valid_results}, got '{result}'"
            )


def validate_training_parameters(**kwargs) -> None:
    """
    Validate training-specific parameters.
    
    Parameters
    ----------
    **kwargs : dict
        Parameters to validate.
        
    Raises
    ------
    ValueError
        If any parameter is invalid.
    """
    # Validate weight range
    if 'wr' in kwargs:
        wr = kwargs['wr']
        if not isinstance(wr, (int, float)) or wr <= 0.0 or wr >= 1.0:
            raise ValueError(
                f"wr must be a float between 0.0 and 1.0, got {wr}"
            )

    # Validate learning rate
    if 'lr' in kwargs:
        lr = kwargs['lr']
        if not isinstance(lr, (int, float)) or lr <= 0.0:
            raise ValueError(
                f"lr must be a positive number, got {lr}"
            )
    
    # Validate epochs
    if 'epochs' in kwargs:
        epochs = kwargs['epochs']
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(
                f"epochs must be a positive integer, got {epochs}"
            )
    
    # Validate k_folds
    if 'k_folds' in kwargs:
        k_folds = kwargs['k_folds']
        if not isinstance(k_folds, int) or k_folds < 2:
            raise ValueError(
                f"k_folds must be an integer >= 2, got {k_folds}"
            )
    
    # Validate test_size
    if 'test_size' in kwargs:
        test_size = kwargs['test_size']
        if not isinstance(test_size, (int, float)) or test_size <= 0.0 or test_size >= 1.0:
            raise ValueError(
                f"test_size must be a float between 0.0 and 1.0, got {test_size}"
            )
    
    # Validate validation_split
    if 'validation_split' in kwargs:
        validation_split = kwargs['validation_split']
        if not isinstance(validation_split, (int, float)) or validation_split <= 0.0 or validation_split >= 1.0:
            raise ValueError(
                f"validation_split must be a float between 0.0 and 1.0, got {validation_split}"
            )
    
    # Validate patience
    if 'patience' in kwargs:
        patience = kwargs['patience']
        if not isinstance(patience, int) or patience <= 0:
            raise ValueError(
                f"patience must be a positive integer, got {patience}"
            )


def validate_data_shape(X: np.ndarray, expected_ndim: int = 2, 
                       min_samples: int = 1, min_features: int = 1) -> None:
    """
    Validate data shape and dimensions.
    
    Parameters
    ----------
    X : ndarray
        Data to validate.
    expected_ndim : int, default=2
        Expected number of dimensions.
    min_samples : int, default=1
        Minimum number of samples.
    min_features : int, default=1
        Minimum number of features.
        
    Raises
    ------
    ValueError
        If data shape is invalid.
    """
    
    if not isinstance(X, np.ndarray):
        raise ValueError(f"X must be a numpy array, got {type(X)}")
    
    if X.ndim != expected_ndim:
        raise ValueError(
            f"X must be {expected_ndim}D array, got {X.ndim}D"
        )
    
    if X.shape[0] < min_samples:
        raise ValueError(
            f"X must have at least {min_samples} samples, got {X.shape[0]}"
        )
    
    if X.shape[1] < min_features:
        raise ValueError(
            f"X must have at least {min_features} features, got {X.shape[1]}"
        )


def validate_binary_classification(y: np.ndarray) -> None:
    """
    Validate that y contains exactly 2 classes.
    
    Parameters
    ----------
    y : ndarray
        Target labels.
        
    Raises
    ------
    ValueError
        If y doesn't contain exactly 2 classes.
    """
    
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        raise ValueError(
            f"Binary classification requires exactly 2 classes, "
            f"found {len(unique_classes)}: {unique_classes}"
        )


def check_quantum_circuit_size(n_qubits: int, max_qubits: int = 20) -> None:
    """
    Check if quantum circuit size is reasonable for simulation.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    max_qubits : int, default=20
        Maximum recommended number of qubits.
        
    Warns
    -----
    UserWarning
        If circuit is too large for efficient simulation.
    """
    
    if n_qubits > max_qubits:
        warnings.warn(
            f"Large quantum circuit with {n_qubits} qubits. "
            f"Simulation may be slow. Consider reducing circuit size or "
            f"using a more powerful quantum simulator.", 
            UserWarning
        )


def validate_random_state(random_state: Optional[Union[int, np.random.RandomState]]) -> None:
    """
    Validate random state parameter.
    
    Parameters
    ----------
    random_state : int, RandomState or None
        Random state to validate.
        
    Raises
    ------
    ValueError
        If random_state is invalid.
    """
    
    if random_state is not None:
        if not isinstance(random_state, (int, np.random.RandomState)):
            raise ValueError(
                f"random_state must be an int, RandomState instance, or None, "
                f"got {type(random_state)}"
            )
        
        if isinstance(random_state, int) and random_state < 0:
            raise ValueError(
                f"random_state must be a non-negative integer, got {random_state}"
            )
