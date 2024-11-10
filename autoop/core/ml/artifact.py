from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import base64
import hashlib


class Artifact(BaseModel):
    """ A class representing an artifact with enhanced functionality. """

    name: Optional[str] = Field(None, description="The name of the artifact.")
    type: Optional[str] = Field(None, description="The type of the artifact.")
    data: Optional[bytes] = Field(
        None,
        description=(
            "The binary data of the artifact."
        ),
    )
    asset_path: Optional[str] = Field(
        None, description="The path to the asset."
    )
    version: Optional[str] = Field(
        None,
        description="The version of the artifact."
    )
    tags: Optional[List[str]] = Field(
        default_factory=list,
        description="List of tags associated with the artifact."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata for the artifact."
    )

    @property
    def id(self) -> str:
        """
        Generates a unique ID for the artifact based on its name and version.
        """
        unique_string = f"{self.name}:{self.version}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def encode_data(self, data: str) -> bytes:
        """
        Encodes a string as base64 bytes and stores it in self.data.

        Args:
            data (str): The string data to encode.

        Returns:
            bytes: The encoded base64 bytes.
        """
        encoded_data = base64.b64encode(data.encode())
        self.data = encoded_data
        return encoded_data

    def decode_data(self) -> str:
        """
        Decodes the base64 bytes in self.data and returns it as a string.

        Returns:
            str: The decoded string data.

        Raises:
            ValueError: If there is no data to decode.
        """
        if self.data is None:
            raise ValueError("No data to decode.")
        return base64.b64decode(self.data).decode()

    def save(self, data: bytes) -> None:
        """
        Saves raw bytes to self.data.

        Args:
            data (bytes): The binary data to save.
        """
        self.data = data

    def read(self) -> bytes:
        """
        Returns raw bytes from self.data.

        Returns:
            bytes: The raw binary data.

        Raises:
            ValueError: If there is no data to read.
        """
        if self.data is None:
            raise ValueError("No data to read.")
        return self.data
