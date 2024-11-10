from typing import List
import pickle
import numpy as np

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features


class Pipeline:

    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_f: Feature,  # Changed from target_feature to target_f
                 split=0.8):
        """
        Initializes the pipeline with the dataset, model, input features,
        target feature, metrics, and split ratio.

        Args:
            metrics: List of Metric objects for evaluating the model.
            dataset: Dataset object containing the data.
            model: Model object to be used in the pipeline.
            input_features: List of input Feature objects.
            target_f: Target Feature object, expected by the tests.
            split: Ratio to split the data into training and testing sets.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_f
        self._metrics = metrics
        self._artifacts = {}
        self._split = split

        # Validation of model type
        if (target_f.feature_type == "categorical" and
                model.type != "classification"):
            raise ValueError(
                "Model type must be classification for categorical target "
                "feature"
            )
        if target_f.feature_type == "numerical" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for numerical target feature"
            )

    def __str__(self):
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self):
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Used to get the artifacts generated during the pipeline execution
        to be saved.

        Returns:
            List of Artifact objects generated during the pipeline execution.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = pickle.dumps(artifact["encoder"])
                artifacts.append(Artifact(name=name, data=data))
            elif artifact_type in ["StandardScaler"]:
                data = pickle.dumps(artifact["scaler"])
                artifacts.append(Artifact(name=name, data=data))

        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        model_artifact = self._model.to_artifact(
            name=f"pipeline_model_{self._model.type}"
        )
        artifacts.append(model_artifact)
        return artifacts

    def _register_artifact(self, name: str, artifact):
        self._artifacts[name] = artifact

    def _preprocess_features(self):
        """
        Preprocesses the features and registers artifacts.
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)

        input_results = preprocess_features(
            self._input_features, self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)

        # Get the input vectors and output vector, sorted by feature name
        # for consistency
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self):
        """
        Splits data into training and testing sets based on the split ratio.
        """
        split_idx = int(self._split * len(self._output_vector))
        self._train_X = [vector[:split_idx] for vector in self._input_vectors]
        self._test_X = [vector[split_idx:] for vector in self._input_vectors]
        self._train_y = self._output_vector[:split_idx]
        self._test_y = self._output_vector[split_idx:]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        return np.column_stack(vectors)

    def _train(self):
        """
        Trains the model using the training data.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)
        # If the model has parameters, store them after training
        if hasattr(self._model, "parameters"):
            self.parameters = self._model.parameters

    def _evaluate(self):
        """
        Evaluates the model on both training and testing data
        and calculates metrics.
        """
        # Evaluate on the test set
        X_test = self._compact_vectors(self._test_X)
        Y_test = self._test_y
        self._test_metrics_results = []
        test_predictions = self._model.predict(X_test)

        for metric in self._metrics:
            result = metric(Y_test, test_predictions)
            self._test_metrics_results.append((metric, result))

        self._test_predictions = test_predictions

        # Evaluate on the training set (not included in _metrics_results
        # to match test expectations)
        X_train = self._compact_vectors(self._train_X)
        Y_train = self._train_y
        self._train_metrics_results = []
        train_predictions = self._model.predict(X_train)

        for metric in self._metrics:
            result = metric(Y_train, train_predictions)
            self._train_metrics_results.append((metric, result))

        self._train_predictions = train_predictions

        # Set _metrics_results to only include test metrics
        # to match the test expectation
        self._metrics_results = self._test_metrics_results
        self._predictions = {
            "train": self._train_predictions,
            "test": self._test_predictions
        }

    def execute(self):
        """
        Executes the full pipeline: preprocesses data, splits data,
        trains and evaluates the model.

        Returns:
            dict: A dictionary with metrics and predictions for both
                  training and testing sets.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()

        return {
            "train_metrics": self._train_metrics_results,
            "train_predictions": self._train_predictions,
            "test_metrics": self._test_metrics_results,
            "test_predictions": self._test_predictions,
        }
