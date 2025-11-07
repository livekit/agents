from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from .log import logger

try:  # pragma: no cover - exercised implicitly when dependency is installed
    from inferedge_moss import (
        DocumentInfo,
        IndexInfo,
        MossClient as _InferEdgeMossClient,
        SearchResult,
    )
except ImportError:  # pragma: no cover - provides clearer message if dependency missing at runtime
    _InferEdgeMossClient = None

    class _MissingDependency:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "inferedge-moss is required to use moss integrations. Install it via `pip install inferedge-moss`."
            )

    class DocumentInfo(_MissingDependency):
        pass

    class IndexInfo(_MissingDependency):
        pass

    class SearchResult(_MissingDependency):
        pass

__all__ = ["MossClient", "DocumentInfo", "IndexInfo", "SearchResult"]


class MossClient:
    """Async helper around :mod:`inferedge_moss` tailored for LiveKit agents."""

    def __init__(
        self,
        project_id: str | None = None,
        project_key: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        if _InferEdgeMossClient is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "inferedge-moss is required to use MossClient. Install it via `pip install inferedge-moss`."
            )

        self._project_id = project_id or os.environ.get("MOSS_PROJECT_ID")
        self._project_key = project_key or os.environ.get("MOSS_PROJECT_KEY")

        if not self._project_id:
            raise ValueError("project_id must be provided or set through the MOSS_PROJECT_ID environment variable")

        if not self._project_key:
            raise ValueError(
                "project_key must be provided or set through the MOSS_PROJECT_KEY environment variable"
            )

        self._client = _InferEdgeMossClient(self._project_id, self._project_key, **client_kwargs)
        self._loaded_indexes: MutableMapping[str, str] = {}
        self._load_lock = asyncio.Lock()

    @property
    def project_id(self) -> str:
        """Return the Moss project identifier used by this client."""

        return self._project_id

    @property
    def project_key(self) -> str:
        """Return the secret project key used for authenticating requests."""

        return self._project_key

    @property
    def inner_client(self) -> Any:
        """Expose the underlying InferEdge client for advanced use cases."""

        return self._client

    async def create_index(
        self,
        index_name: str,
        documents: Sequence[DocumentInfo],
        model_id: str,
    ) -> bool:
        """Create a new Moss index and populate it with ``documents``."""

        logger.debug("creating moss index", extra={"index": index_name, "model": model_id})
        result = await self._client.create_index(index_name, documents, model_id)
        return bool(result)

    async def load_index(self, index_name: str, *, force: bool = False) -> str:
        """Load ``index_name`` into the local cache, returning the storage path."""

        if not force and index_name in self._loaded_indexes:
            return self._loaded_indexes[index_name]

        async with self._load_lock:
            if not force and index_name in self._loaded_indexes:
                return self._loaded_indexes[index_name]

            logger.debug("loading moss index", extra={"index": index_name})
            path = await self._client.load_index(index_name)
            if isinstance(path, str):
                self._loaded_indexes[index_name] = path
            else:
                logger.debug("unexpected load_index return type", extra={"type": type(path).__name__})
            return self._loaded_indexes.get(index_name, "")

    async def query(
        self,
        index_name: str,
        query: str,
        top_k: int = 5,
        *,
        auto_load: bool = True,
    ) -> SearchResult:
        """Execute a semantic search against ``index_name`` and return results."""

        if auto_load and index_name not in self._loaded_indexes:
            await self.load_index(index_name)
        elif not auto_load and index_name not in self._loaded_indexes:
            raise RuntimeError(
                f"index '{index_name}' is not loaded. Call load_index() first or enable auto_load."
            )

        logger.debug("querying moss index", extra={"index": index_name, "top_k": top_k})
        return await self._client.query(index_name, query, top_k)

    async def add_documents(
        self,
        index_name: str,
        documents: Sequence[DocumentInfo],
    ) -> Mapping[str, int]:
        """Append ``documents`` to ``index_name`` and return the backend counts."""

        logger.debug("adding documents to moss index", extra={"index": index_name, "count": len(documents)})
        mapping = await self._client.add_docs(index_name, documents)
        if not isinstance(mapping, Mapping):
            raise TypeError("inferedge_moss.add_docs returned an unexpected type")
        return mapping

    async def list_indexes(self) -> Sequence[IndexInfo]:
        """Return every index visible in the configured Moss project."""

        logger.debug("listing moss indexes")
        indexes = await self._client.list_indexes()
        if not isinstance(indexes, Sequence):
            raise TypeError("inferedge_moss.list_indexes returned an unexpected type")
        return indexes

    async def delete_index(self, index_name: str) -> bool:
        """Remove ``index_name`` remotely and evict it from the local cache."""

        logger.debug("deleting moss index", extra={"index": index_name})
        deleted = await self._client.delete_index(index_name)
        self._loaded_indexes.pop(index_name, None)
        return bool(deleted)

    def is_index_loaded(self, index_name: str) -> bool:
        """Return ``True`` if ``index_name`` is present in the local cache."""

        return index_name in self._loaded_indexes

    def unload_index(self, index_name: str) -> None:
        """Drop any cached information for ``index_name`` without API calls."""

        self._loaded_indexes.pop(index_name, None)

    def __repr__(self) -> str:
        return f"MossClient(project_id={self._project_id!r})"
