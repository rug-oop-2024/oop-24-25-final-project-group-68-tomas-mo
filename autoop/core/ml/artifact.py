from pydantic import BaseModel, Field
from datetime import datetime
import base64
from typing import Optional


class Artifact(BaseModel):
    """
    A class representing an artifact in a machine learning pipeline, 
    typically used to store models, data files, or other relevant objects.
    """
    id: Optional[str] = Field(None, description="Unique identifier for the artifact")
    name: str = Field(..., description="Name of the artifact")
    data: Optional[bytes] = Field(None, description="Binary data associated with the artifact")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of artifact creation")

    def encode_data(self):
        """
        Encodes the artifact's data into a base64 string.
        
        Returns:
            str: Base64 encoded string of the artifact's data.
        """
        if self.data:
            return base64.b64encode(self.data).decode('utf-8')
        return None

    def decode_data(self, encoded_data: str):
        """
        Decodes a base64 string back to binary data and assigns it to the artifact.
        
        Args:
            encoded_data (str): Base64 encoded string of data to decode.
        """
        self.data = base64.b64decode(encoded_data.encode('utf-8'))

    def save_to_file(self, file_path: str):
        """
        Saves the binary data of the artifact to a file.
        
        Args:
            file_path (str): Path where the file should be saved.
        """
        if self.data:
            with open(file_path, 'wb') as file:
                file.write(self.data)

    @classmethod
    def load_from_file(cls, file_path: str, name: str) -> 'Artifact':
        """
        Loads binary data from a file and creates an artifact instance.

        Args:
            file_path (str): Path to the file to load.
            name (str): Name for the artifact instance.

        Returns:
            Artifact: An instance of the Artifact class with loaded data.
        """
        with open(file_path, 'rb') as file:
            data = file.read()
        return cls(name=name, data=data)
