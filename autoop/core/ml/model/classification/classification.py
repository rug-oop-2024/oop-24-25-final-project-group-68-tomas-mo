from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from model import Model  # Import the base Model class


class LogisticRegressionModel(Model):
    def __init__(self):
        """Initialize a logistic regression model using scikit-learn."""
        self.model = LogisticRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on the provided data."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the provided data."""
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the accuracy score of the model on the provided data."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class DecisionTreeClassification(Model):
    def __init__(self):
        """Initialize a decision tree classifier using scikit-learn."""
        self.model = DecisionTreeClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on the provided data."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the provided data."""
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the accuracy score of the model on the provided data."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class RandomForestClassification(Model):
    def __init__(self):
        """Initialize a random forest classifier using scikit-learn."""
        self.model = RandomForestClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on the provided data."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the provided data."""
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the accuracy score of the model on the provided data."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class SVCClassification(Model):
    def __init__(self):
        """Initialize a Support Vector Classifier (SVC) using scikit-learn."""
        self.model = SVC()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on the provided data."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the provided data."""
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the accuracy score of the model on the provided data."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
