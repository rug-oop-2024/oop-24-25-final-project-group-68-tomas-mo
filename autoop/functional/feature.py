from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detects feature types in a dataset.

    Assumptions:
        - Only numerical and categorical types are considered.
        - The dataset does not contain NaN values.

    Args:
        dataset (Dataset): The dataset containing the data.

    Returns:
        List[Feature]: A list of Feature objects, each with a name and detected type.
    """
    features = []
    for column in dataset.data.columns:
        # Detect feature type based on data type
        if dataset.data[column].dtype in ['int64', 'float64']:
            feature_type = 'numerical'
        else:
            feature_type = 'categorical'

        # Append Feature object with detected type
        features.append(Feature(name=column, feature_type=feature_type))

    return features
