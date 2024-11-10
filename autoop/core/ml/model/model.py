from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class Model(ABC):
    type: str

    def __init__(self, **params):
        """Initialize model with hyperparameters."""
        self.params = params

    @abstractmethod
    def fit(self, X, y):
        """Train the model with input data X and target y."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict using the trained model on input data X."""
        pass

    def evaluate(self, X, y, metric):
        """Evaluate the model using a specified metric."""
        y_pred = self.predict(X)
        return metric(y, y_pred)


# Classification Models

class LogisticRegressionModel(Model):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = LogisticRegression(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class DecisionTreeModel(Model):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = DecisionTreeClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class RandomForestModel(Model):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = RandomForestClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


# Regression Models

class LinearRegressionModel(Model):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = LinearRegression(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class DecisionTreeRegressorModel(Model):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = DecisionTreeRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class RandomForestRegressorModel(Model):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = RandomForestRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
