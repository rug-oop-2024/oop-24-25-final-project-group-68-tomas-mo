from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List, Optional


class ArtifactRegistry:
    def __init__(self, database: Database, storage: Storage) -> None:
        """
        Initialize an ArtifactRegistry with a database and storage backend.

        Args:
            database (Database): The database to store artifact metadata.
            storage (Storage): The storage to save artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Register an artifact by saving it in storage and
          recording its metadata in the database.

        Args:
            artifact (Artifact): The artifact to register.
        """
        self._storage.save(artifact.data, artifact.asset_path)
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: Optional[str] = None) -> List[Artifact]:
        """
        List all artifacts, optionally filtering by type.

        Args:
            type (Optional[str]): Filter artifacts by type if provided.

        Returns:
            List[Artifact]: A list of Artifact objects that
              match the specified type.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieve an artifact by its ID.

        Args:
            artifact_id (str): The unique identifier for the artifact.

        Returns:
            Artifact: The artifact object corresponding to the specified ID.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Delete an artifact by its ID, removing it from storage
          and the database.

        Args:
            artifact_id (str): The unique identifier for the
              artifact to delete.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initialize the AutoML system with storage and database backends.

        Args:
            storage (LocalStorage): The storage backend for artifacts.
            database (Database): The database backend for storing metadata.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Get a singleton instance of the AutoMLSystem.

        Returns:
            AutoMLSystem: The singleton instance of the AutoMLSystem.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(LocalStorage("./assets/dbo")),
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Access the artifact registry.

        Returns:
            ArtifactRegistry: The registry for managing artifacts.
        """
        return self._registry
