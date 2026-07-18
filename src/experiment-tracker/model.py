from sklearn.metrics import accuracy_score
from .decorator import timer, log_errors, retry
from abc import ABC, abstractmethod  

class BaseModel(ABC):
    """Abstract base class all model wrappers should extend."""

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class LogisticRegression(BaseModel):
    """Wraps a scikit-learn Logistic Regression classifier with train/evaluate/predict methods."""

    def __init__(self, **model_params):
        from sklearn.linear_model import LogisticRegression

        self.model_params = model_params
        self.model = LogisticRegression(**model_params)
        self.is_trained = False

    @timer
    @log_errors
    @retry(n=3)
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self.model

    def evaluate(self, X_test, y_test) -> float:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")
        preds = self.predict(X_test)
        return accuracy_score(y_test, preds)

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X)

class RandomForest(BaseModel):
    """Wraps a scikit-learn Random Forest classifier with train/evaluate/predict methods."""

    def __init__(self, **model_params):
        from sklearn.ensemble import RandomForestClassifier
        
        self.model_params = model_params
        self.model = RandomForestClassifier(**model_params)
        self.is_trained = False

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self.model

    def evaluate(self, X_test, y_test) -> float:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")
        preds = self.predict(X_test)
        return accuracy_score(y_test, preds)

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X)