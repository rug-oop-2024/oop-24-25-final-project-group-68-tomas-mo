from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from autoop.core.ml.model.model import Model  # Import the base Model class


class LogisticRegressionModel(Model):
    def __init__(self) -> None:
        """
        Initialize a logistic regression model using scikit-learn.
        """
        self.model = LogisticRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model on the provided data.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target labels.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the provided data using the trained logistic
          regression model.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the accuracy score of the logistic regression model on
          the provided data.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): True labels.

        Returns:
            float: Accuracy score of the model.
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class DecisionTreeClassification(Model):
    def __init__(self) -> None:
        """
        Initialize a decision tree classifier using scikit-learn.
        """
        self.model = DecisionTreeClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the decision tree classifier on the provided data.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target labels.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the provided data using the trained decision
          tree classifier.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the accuracy score of the decision tree classifier on the
          provided data.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): True labels.

        Returns:
            float: Accuracy score of the model.
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class RandomForestClassification(Model):
    def __init__(self) -> None:
        """
        Initialize a random forest classifier using scikit-learn.
        """
        self.model = RandomForestClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the random forest classifier on the provided data.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target labels.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the provided data using the trained random forest
          classifier.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the accuracy score of the random forest classifier on the
          provided data.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): True labels.

        Returns:
            float: Accuracy score of the model.
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
