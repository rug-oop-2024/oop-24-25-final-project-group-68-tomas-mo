import numpy as np
from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def __call__(self, y_true, y_pred):
        """Compute the metric given true and predicted values."""
        pass


# Classification Metrics

class Accuracy(Metric):
    def __call__(self, y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        return correct / total if total > 0 else 0


class Precision(Metric):
    def __call__(self, y_true, y_pred):
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / predicted_positives if predicted_positives != 0 else 0


class Recall(Metric):
    def __call__(self, y_true, y_pred):
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / actual_positives if actual_positives != 0 else 0


# Regression Metrics

class MeanSquaredError(Metric):
    def __call__(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


class MeanAbsoluteError(Metric):
    def __call__(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))


class R2Score(Metric):
    def __call__(self, y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_res / ss_total) if ss_total != 0 else 0
