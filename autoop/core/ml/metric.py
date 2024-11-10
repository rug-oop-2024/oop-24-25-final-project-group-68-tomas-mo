import numpy as np
from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the metric given true and predicted values.

        Args:
            y_true (np.ndarray): True labels or target values.
            y_pred (np.ndarray): Predicted labels or target values.

        Returns:
            float: The computed metric value.
        """
        pass


# Classification Metrics

class Accuracy(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the accuracy of predictions.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: The accuracy score, calculated
              as the proportion of correct predictions.
        """
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        return correct / total if total > 0 else 0


class Precision(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the precision of predictions.

        Precision is the ratio of true positives to
          the sum of true and false positives.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: The precision score.
        """
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        return (true_positives / predicted_positives
                if predicted_positives != 0 else 0)


class Recall(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the recall of predictions.

        Recall is the ratio of true positives to the sum
          of true positives and false negatives.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: The recall score.
        """
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return (true_positives / actual_positives
                if actual_positives != 0 else 0)


# Regression Metrics

class MeanSquaredError(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the Mean Squared Error (MSE) of predictions.

        MSE measures the average of the squared differences
          between actual and predicted values.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            float: The MSE score.
        """
        return np.mean((y_true - y_pred) ** 2)


class MeanAbsoluteError(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the Mean Absolute Error (MAE) of predictions.

        MAE measures the average of the absolute differences between
          actual and predicted values.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            float: The MAE score.
        """
        return np.mean(np.abs(y_true - y_pred))


class R2Score(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the R-squared (R²) score of predictions.

        R² represents the proportion of the variance in the dependent
          variable that is predictable from the independent variables.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            float: The R² score, a measure of goodness-of-fit. The score
              ranges from 0 to 1, where 1 indicates perfect prediction.
        """
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_res / ss_total) if ss_total != 0 else 0
