import abc
import numpy as np
from typing import Any, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets


class QuantumBaseModel(BaseEstimator, ClassifierMixin):
    """
    Base class for quantum machine learning models following scikit-learn conventions.
    
    This class provides a common interface for all quantum models and ensures
    compatibility with scikit-learn's ecosystem.
    
    Parameters
    ----------
    lr : float, default=0.1
        Learning rate for gradient-based optimizers.
    epochs : int, default=50
        Maximum number of training iterations.
    optimize : bool, default=True
        Whether to perform hyperparameter optimization during fit.
    random_state : int or None, default=None
        Random seed for reproducible results.
    """
    
    def __init__(self, lr: float = 0.1, epochs: int = 50, 
                 optimize: bool = True, random_state: Optional[int] = None):
        self.lr = lr
        self.epochs = epochs
        self.optimize = optimize
        self.random_state = random_state
        
        self.params_ = None
        self.history_ = []
        self.n_features_in_ = None
        self.classes_ = None
        self.n_classes_ = None
        
    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumBaseModel':
        """
        Fit the quantum model to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (class labels).
            
        Returns
        -------
        self : QuantumBaseModel
            Returns self for method chaining.
        """
        pass
    
    @abc.abstractmethod
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
        pass
    
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

        raise NotImplementedError("predict_proba must be implemented by subclasses")
    
    def score(self, X: np.ndarray, y: np.ndarray, 
              sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    
    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                       reset: bool = False) -> tuple:
        """
        Validate input data following scikit-learn conventions.
        
        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like, optional
            Target data.
        reset : bool, default=False
            Whether to reset fitted attributes.
            
        Returns
        -------
        X_validated : ndarray
            Validated input data.
        y_validated : ndarray, optional
            Validated target data.
        """
        if reset:
            self.n_features_in_ = None
            self.classes_ = None
            self.n_classes_ = None
            
        if y is not None:
            X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
            check_classification_targets(y)

            self.classes_, y_indices = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
            self.n_features_in_ = X.shape[1]
            
            return X, y_indices
        else:
            X = check_array(X, accept_sparse=False, dtype=np.float64)
            if hasattr(self, 'n_features_in_') and self.n_features_in_ is not None:
                if X.shape[1] != self.n_features_in_:
                    raise ValueError(
                        f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                        f"is expecting {self.n_features_in_} features as input."
                    )
            return X, None
    
    def _check_is_fitted(self) -> None:
        """Check if the estimator is fitted."""
        if not hasattr(self, 'params_') or self.params_ is None:
            raise ValueError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )