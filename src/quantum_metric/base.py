import abc

class QuantumBaseModel(abc.ABC):
    """Clase base para modelos cuánticos estilo scikit-learn."""

    def __init__(self, lr=0.1, epochs=50, optimize=True):
        self.lr = lr
        self.epochs = epochs
        self.optimize = optimize
        self.params_ = None  # al estilo sklearn (termina con _)
        self.history_ = []   # guardar evolución del entrenamiento

    @abc.abstractmethod
    def fit(self, X, y):
        """Entrena el modelo sobre los datos (debe implementarse)."""
        pass

    @abc.abstractmethod
    def predict(self, X):
        """Genera predicciones para nuevos datos (debe implementarse)."""
        pass

    def score(self, X, y, metric=None):
        """Calcula una métrica de desempeño."""
        preds = self.predict(X)
        if metric is not None:
            return metric(y, preds)
        return None

    def get_params(self):
        """Devuelve los hiperparámetros del modelo."""
        return {
            "lr": self.lr,
            "epochs": self.epochs,
            "optimize": self.optimize
        }

    def set_params(self, **params):
        """Actualiza los hiperparámetros del modelo."""
        for key, value in params.items():
            setattr(self, key, value)
        return self