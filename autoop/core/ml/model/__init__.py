from autoop.core.ml.model.model import Model
from autoop.core.ml.model.classification import (
    LogisticRegressionModel,
    DecisionTreeClassification,
    RandomForestClassification
)

from autoop.core.ml.model.regression import (
    MultipleLinearRegression,
    DecisionTreeRegression,
    RandomForestRegression,
    SVRRegression
)

REGRESSION_MODELS = [
    "Multiple Linear Regression",
    "Decision Tree Regression",
    "Random Forest Regression",
    "SVR Regression"
]

CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Decision Tree Classification",
    "Random Forest Classification"
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name == "Logistic Regression":
        return LogisticRegressionModel()
    elif model_name == "Decision Tree Classifier":
        return DecisionTreeClassification()
    elif model_name == "Random Forest Classifier":
        return RandomForestClassification()
    elif model_name == "Multiple Linear Regression":
        return MultipleLinearRegression()
    elif model_name == "Decision Tree Regression":
        return DecisionTreeRegression()
    elif model_name == "Random Forest Regression":
        return RandomForestRegression()
    elif model_name == "SVR Regression":
        return SVRRegression()
    else:
        raise ValueError(f"Model '{model_name}' isn't recognised")
