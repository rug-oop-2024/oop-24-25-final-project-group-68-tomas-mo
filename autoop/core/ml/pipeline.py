from typing import List, Dict, Any
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
                 target_f: Feature,
                 split: float = 0.8):
        """
        Initializes the pipeline with dataset, model, input features,
        target feature, metrics, and split ratio.

        Args:
            metrics (List[Metric]): List of Metric objects to evaluate
            the model.
            dataset (Dataset): Dataset object containing the data.
            model (Model): Model object used in the pipeline.
            input_features (List[Feature]): List of input Feature objects.
            target_f (Feature): Target Feature object.
            split (float): Ratio to split data into training and testing sets.
        """
        self._dataset: Dataset = dataset
        self._model: Model = model
        self._input_features: List[Feature] = input_features
        self._target_feature: Feature = target_f
        self._metrics: List[Metric] = metrics
        self._artifacts: Dict[str, Any] = {}
        self._split: float = split

        if (target_f.feature_type == "categorical" and
                model.type != "classification"):
            raise ValueError(
                "Model type must be classification for categorical target "
                "feature"
            )
        if target_f.feature_type == "numerical" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for numerical target "
                "feature"
            )

    def __str__(self) -> str:
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
    def model(self) -> Model:
        """Returns the model used in the pipeline."""
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Retrieves artifacts generated during pipeline execution.

        Returns:
            List[Artifact]: List of artifacts generated during pipeline
              execution.
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

    def _register_artifact(self, name: str, artifact: Any) -> None:
        """
        Registers an artifact.

        Args:
            name (str): Name of the artifact.
            artifact (Any): The artifact to register.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocesses the features and registers artifacts.
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)

        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)

        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self) -> None:
        """
        Splits data into training and testing sets based on the split ratio.
        """
        split_idx = int(self._split * len(self._output_vector))
        self._train_X = [vector[:split_idx] for vector in self._input_vectors]
        self._test_X = [vector[split_idx:] for vector in self._input_vectors]
        self._train_y = self._output_vector[:split_idx]
        self._test_y = self._output_vector[split_idx:]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Combines multiple arrays into a single array by stacking columns.

        Args:
            vectors (List[np.array]): List of numpy arrays.

        Returns:
            np.array: Combined array.
        """
        return np.column_stack(vectors)

    def _train(self) -> None:
        """
        Trains the model using the training data.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)
        if hasattr(self._model, "parameters"):
            self.parameters = self._model.parameters

    def _evaluate(self) -> None:
        """
        Evaluates the model on training and testing data and calculates
          metrics.
        """
        X_test = self._compact_vectors(self._test_X)
        Y_test = self._test_y
        self._test_metrics_results = []
        test_predictions = self._model.predict(X_test)

        for metric in self._metrics:
            result = metric(Y_test, test_predictions)
            self._test_metrics_results.append((metric, result))

        self._test_predictions = test_predictions

        X_train = self._compact_vectors(self._train_X)
        Y_train = self._train_y
        self._train_metrics_results = []
        train_predictions = self._model.predict(X_train)

        for metric in self._metrics:
            result = metric(Y_train, train_predictions)
            self._train_metrics_results.append((metric, result))

        self._train_predictions = train_predictions
        self._metrics_results = self._test_metrics_results
        self._predictions = {
            "train": self._train_predictions,
            "test": self._test_predictions
        }

    def execute(self) -> Dict[str, Any]:
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
