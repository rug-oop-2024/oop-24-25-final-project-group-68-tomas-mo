from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional
import pandas as pd


class Feature(BaseModel):
    """
    Represents a feature in a dataset, with associated metadata and methods 
    to compute and retrieve statistics.
    """
    name: str = Field(..., description="Name of the feature")
    feature_type: Literal['numerical', 'categorical'] = Field(..., description="Type of the feature")
    description: Optional[str] = Field(None, description="Description of the feature")
    data: Optional[pd.Series] = Field(None, description="Data series associated with the feature")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __str__(self) -> str:
        """
        Provides a string representation of the feature, displaying its name and type.
        
        Returns:
            str: A string representation of the feature.
        """
        return f"Feature(name={self.name}, type={self.feature_type})"

    def compute_statistics(self) -> Optional[dict]:
        """
        Computes statistics based on the feature type.

        Returns:
            dict: A dictionary of statistics (mean, median, std_dev for numerical; 
                  mode and value counts for categorical).
        """
        if self.data is None:
            return None

        if self.feature_type == 'numerical':
            return {
                'mean': self.data.mean(),
                'median': self.data.median(),
                'std_dev': self.data.std()
            }
        elif self.feature_type == 'categorical':
            return {
                'mode': self.data.mode().iloc[0] if not self.data.mode().empty else None,
                'value_counts': self.data.value_counts().to_dict()
            }
        return None
