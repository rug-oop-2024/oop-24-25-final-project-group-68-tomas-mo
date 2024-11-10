from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
from autoop.core.ml.model.model import Model  # Import the base Model class


class MultipleLinearRegression(Model):
    def __init__(self):
        """Initialize a multiple linear regression model using scikit-learn."""
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on the provided data."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the provided data."""
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the R^2 score of the model on the provided data."""
        return self.model.score(X, y)


class DecisionTreeRegression(Model):
    def __init__(self):
        """Initialize a decision tree regression model using scikit-learn."""
        self.model = DecisionTreeRegressor()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on the provided data."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the provided data."""
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the R^2 score of the model on the provided data."""
        return self.model.score(X, y)


class RandomForestRegression(Model):
    def __init__(self):
        """Initialize a random forest regression model using scikit-learn."""
        self.model = RandomForestRegressor()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on the provided data."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the provided data."""
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the R^2 score of the model on the provided data."""
        return self.model.score(X, y)


class SVRRegression(Model):
    def __init__(self):
        """Initialize a Support Vector Regression (SVR) model using scikit-learn."""
        self.model = SVR()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on the provided data."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the provided data."""
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the R^2 score of the model on the provided data."""
        return self.model.score(X, y)
