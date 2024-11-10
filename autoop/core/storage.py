from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    def __init__(self, path: str) -> None:
        """
        Exception raised when a specified path is not found.

        Args:
            path (str): The path that was not found.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a specified path.

        Args:
            data (bytes): The data to save.
            path (str): The path where data will be saved.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a specified path.

        Args:
            path (str): The path to load data from.

        Returns:
            bytes: The loaded data.
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a specified path.

        Args:
            path (str): The path to delete data from.
        """
        pass

    @abstractmethod
    def list(self, path: str) -> List[str]:
        """
        List all file paths under a specified path.

        Args:
            path (str): The path to list files from.

        Returns:
            List[str]: List of file paths under the specified path.
        """
        pass


class LocalStorage(Storage):
    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initialize a LocalStorage instance with a base directory path.

        Args:
            base_path (str): The base directory path where files
            will be stored.
                             Defaults to "./assets".
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to a file at the specified key path.

        Args:
            data (bytes): The data to save.
            key (str): The key (relative path) under the
            base path to save data.
        """
        path = self._join_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a file at the specified key path.

        Args:
            key (str): The key (relative path) under the
            base path to load data.

        Returns:
            bytes: The loaded data.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete a file at the specified key path.

        Args:
            key (str): The key (relative path) under the
            base path to delete data.
                       Defaults to "/".
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        List all files under the specified prefix path.

        Args:
            prefix (str): The prefix (relative path) to list files from.
                          Defaults to "/".

        Returns:
            List[str]: A list of relative file paths under the
              specified prefix.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [
            os.path.relpath(p, self._base_path)
            for p in keys if os.path.isfile(p)
        ]

    def _assert_path_exists(self, path: str) -> None:
        """
        Check if the specified path exists, raising NotFoundError if it
          does not.

        Args:
            path (str): The path to check for existence.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Join the base path with the specified path to create an absolute path.

        Args:
            path (str): The relative path to join with the base path.

        Returns:
            str: The joined, normalized absolute path.
        """
        return os.path.normpath(os.path.join(self._base_path, path))
