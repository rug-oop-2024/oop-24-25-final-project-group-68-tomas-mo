from typing import Any, Dict
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np


class Model(ABC):
    type: str

    def __init__(self, **params: Any) -> None:
        """
        Initializes the model with hyperparameters.

        Args:
            **params: Hyperparameters to initialize the model.
        """
        self.params: Dict[str, Any] = params

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the model using the input data and target labels.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target labels.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts outcomes using the trained model on input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted outcomes.
        """
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric: Any) -> float:
        """
        Evaluates the model using a specified metric.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): True target labels.
            metric (callable): A metric function to evaluate predictions.

        Returns:
            float: Metric value calculated on predictions.
        """
        y_pred = self.predict(X)
        return metric(y, y_pred)


# Classification Models

class LogisticRegressionModel(Model):
    def __init__(self, **params: Any) -> None:
        """
        Initializes a Logistic Regression model with specified parameters.

        Args:
            **params: Hyperparameters for Logistic Regression.
        """
        super().__init__(**params)
        self.model = LogisticRegression(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the Logistic Regression model.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target labels.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts outcomes using the trained Logistic Regression model.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted outcomes.
        """
        return self.model.predict(X)


class DecisionTreeModel(Model):
    def __init__(self, **params: Any) -> None:
        """
        Initializes a Decision Tree Classifier model with specified parameters.

        Args:
            **params: Hyperparameters for Decision Tree Classifier.
        """
        super().__init__(**params)
        self.model = DecisionTreeClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the Decision Tree Classifier model.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target labels.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts outcomes using the trained Decision Tree Classifier.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted outcomes.
        """
        return self.model.predict(X)


class RandomForestModel(Model):
    def __init__(self, **params: Any) -> None:
        """
        Initializes a Random Forest Classifier model with specified parameters.

        Args:
            **params: Hyperparameters for Random Forest Classifier.
        """
        super().__init__(**params)
        self.model = RandomForestClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the Random Forest Classifier model.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target labels.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts outcomes using the trained Random Forest Classifier.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted outcomes.
        """
        return self.model.predict(X)


# Regression Models

class LinearRegressionModel(Model):
    def __init__(self, **params: Any) -> None:
        """
        Initializes a Linear Regression model with specified parameters.

        Args:
            **params: Hyperparameters for Linear Regression.
        """
        super().__init__(**params)
        self.model = LinearRegression(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the Linear Regression model.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target variable.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts outcomes using the trained Linear Regression model.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted outcomes.
        """
        return self.model.predict(X)


class DecisionTreeRegressorModel(Model):
    def __init__(self, **params: Any) -> None:
        """
        Initializes a Decision Tree Regressor model with specified parameters.

        Args:
            **params: Hyperparameters for Decision Tree Regressor.
        """
        super().__init__(**params)
        self.model = DecisionTreeRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the Decision Tree Regressor model.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target variable.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts outcomes using the trained Decision Tree Regressor.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted outcomes.
        """
        return self.model.predict(X)


class RandomForestRegressorModel(Model):
    def __init__(self, **params: Any) -> None:
        """
        Initializes a Random Forest Regressor model with specified parameters.

        Args:
            **params: Hyperparameters for Random Forest Regressor.
        """
        super().__init__(**params)
        self.model = RandomForestRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the Random Forest Regressor model.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target variable.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts outcomes using the trained Random Forest Regressor.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted outcomes.
        """
        return self.model.predict(X)
