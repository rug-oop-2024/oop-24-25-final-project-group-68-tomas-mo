import json
from typing import Tuple, List, Union, Dict
import os

from autoop.core.storage import Storage


class Database:
    def __init__(self, storage: Storage) -> None:
        """
        Initialize the Database with a storage backend.

        Args:
            storage (Storage): A storage backend implementing the
              Storage interface.
        """
        self._storage = storage
        self._data: Dict[str, Dict[str, dict]] = {}
        self._load()

    def set(self, collection: str, id: str, entry: dict) -> dict:
        """
        Store an entry in the specified collection with a given ID.

        Args:
            collection (str): The collection to store the data in.
            id (str): The unique identifier for the entry.
            entry (dict): The data to store.

        Returns:
            dict: The data that was stored.
        """
        assert isinstance(entry, dict), "Data must be a dictionary"
        assert isinstance(collection, str), "Collection must be a string"
        assert isinstance(id, str), "ID must be a string"
        if not self._data.get(collection, None):
            self._data[collection] = {}
        self._data[collection][id] = entry
        self._persist()
        return entry

    def get(self, collection: str, id: str) -> Union[dict, None]:
        """
        Retrieve an entry from the specified collection by ID.

        Args:
            collection (str): The collection to get the data from.
            id (str): The unique identifier for the entry.

        Returns:
            Union[dict, None]: The data stored with the given ID, or None
              if not found.
        """
        if not self._data.get(collection, None):
            return None
        return self._data[collection].get(id, None)

    def delete(self, collection: str, id: str) -> None:
        """
        Delete an entry from the specified collection by ID.

        Args:
            collection (str): The collection to delete the data from.
            id (str): The unique identifier for the entry.

        Returns:
            None
        """
        if not self._data.get(collection, None):
            return
        if self._data[collection].get(id, None):
            del self._data[collection][id]
        self._persist()

    def list(self, collection: str) -> List[Tuple[str, dict]]:
        """
        List all entries in the specified collection.

        Args:
            collection (str): The collection to list entries from.

        Returns:
            List[Tuple[str, dict]]: A list of tuples containing the ID and data
            for each entry in the collection.
        """
        if not self._data.get(collection, None):
            return []
        return [(id, data) for id, data in self._data[collection].items()]

    def refresh(self) -> None:
        """
        Reload the data from storage to refresh the database state.
        """
        self._load()

    def _persist(self) -> None:
        """
        Persist the current database state to storage, saving or deleting
          entries as necessary.
        """
        for collection, data in self._data.items():
            if not data:
                continue
            for id, item in data.items():
                self._storage.save(
                    json.dumps(item).encode(), f"{collection}{os.sep}{id}"
                )

        # Remove entries in storage that have been deleted from the database
        keys = self._storage.list("")
        for key in keys:
            collection, id = key.split(os.sep)[-2:]
            if not self._data.get(collection, {}).get(id):
                self._storage.delete(f"{collection}{os.sep}{id}")

    def _load(self) -> None:
        """
        Load the data from storage and populate the database.
        """
        self._data = {}
        for key in self._storage.list(""):
            collection, id = key.split(os.sep)[-2:]
            data = self._storage.load(f"{collection}{os.sep}{id}")
            if collection not in self._data:
                self._data[collection] = {}
            self._data[collection][id] = json.loads(data.decode())
