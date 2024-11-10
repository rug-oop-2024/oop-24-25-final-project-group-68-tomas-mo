from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
from autoop.core.ml.model.model import Model


class MultipleLinearRegression(Model):
    def __init__(self) -> None:
        """
        Initialize a multiple linear regression model using scikit-learn.
        """
        self.type = "regression"
        self.model = LinearRegression()
        self.parameters = None  # Initialize parameters attribute

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the provided data and store the model coefficients
        as parameters.

        Args:
            X (np.ndarray): Input feature matrix with shape
              (n_samples, n_features).
            y (np.ndarray): Target variable array with shape (n_samples,).
        """
        self.model.fit(X, y)
        self.parameters = {
            "coefficients": self.model.coef_,
            "intercept": self.model.intercept_
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the provided data.

        Args:
            X (np.ndarray): Input feature matrix with shape
              (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values with shape (n_samples,).
        """
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the R^2 score of the model on the provided data.

        Args:
            X (np.ndarray): Input feature matrix with shape
              (n_samples, n_features).
            y (np.ndarray): Target variable array with shape (n_samples,).

        Returns:
            float: R^2 score of the model on the given data.
        """
        return self.model.score(X, y)


class DecisionTreeRegression(Model):
    def __init__(self) -> None:
        """
        Initialize a decision tree regression model using scikit-learn.
        """
        self.type = "regression"
        self.model = DecisionTreeRegressor()
        self.parameters = None  # Decision trees do not have coefficients

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the provided data.

        Args:
            X (np.ndarray): Input feature matrix with shape
              (n_samples, n_features).
            y (np.ndarray): Target variable array with shape (n_samples,).
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the provided data.

        Args:
            X (np.ndarray): Input feature matrix with shape
              (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values with shape (n_samples,).
        """
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the R^2 score of the model on the provided data.

        Args:
            X (np.ndarray): Input feature matrix with shape
              (n_samples, n_features).
            y (np.ndarray): Target variable array with shape (n_samples,).

        Returns:
            float: R^2 score of the model on the given data.
        """
        return self.model.score(X, y)


class RandomForestRegression(Model):
    def __init__(self) -> None:
        """
        Initialize a random forest regression model using scikit-learn.
        """
        self.type = "regression"
        self.model = RandomForestRegressor()
        self.parameters = None  # Random forests do not have coefficients

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the provided data.

        Args:
            X (np.ndarray): Input feature matrix with shape
              (n_samples, n_features).
            y (np.ndarray): Target variable array with shape (n_samples,).
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the provided data.

        Args:
            X (np.ndarray): Input feature matrix with shape
              (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values with shape (n_samples,).
        """
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the R^2 score of the model on the provided data.

        Args:
            X (np.ndarray): Input feature matrix with shape
              (n_samples, n_features).
            y (np.ndarray): Target variable array with shape (n_samples,).

        Returns:
            float: R^2 score of the model on the given data.
        """
        return self.model.score(X, y)


class SVRRegression(Model):
    def __init__(self) -> None:
        """
        Initialize a Support Vector Regression (SVR) model using scikit-learn.
        """
        self.type = "regression"
        self.model = SVR()
        self.parameters = None  # SVR does not have standard coefficients

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the provided data.

        Args:
            X (np.ndarray): Input feature matrix with shape
              (n_samples, n_features).
            y (np.ndarray): Target variable array with shape (n_samples,).
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the provided data.

        Args:
            X (np.ndarray): Input feature matrix with shape
              (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values with shape (n_samples,).
        """
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the R^2 score of the model on the provided data.

        Args:
            X (np.ndarray): Input feature matrix with shape
              (n_samples, n_features).
            y (np.ndarray): Target variable array with shape (n_samples,).

        Returns:
            float: R^2 score of the model on the given data.
        """
        return self.model.score(X, y)
