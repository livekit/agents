from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inferedge_moss import (
        AddDocumentsOptions,
        DocumentInfo,
        GetDocumentsOptions,
        IndexInfo,
        MossClient as Client,
        QueryOptions,
        SearchResult,
    )
else:
    try:
        from inferedge_moss import (
            AddDocumentsOptions,
            DocumentInfo,
            GetDocumentsOptions,
            IndexInfo,
            MossClient as Client,
            QueryOptions,
            SearchResult,
        )
    except ImportError:
        Client = None  # type: ignore[misc, assignment]

        class _MissingDependency:
            def __init__(self, *_: Any, **__: Any) -> None:
                raise RuntimeError(
                    "inferedge-moss is required to use moss integrations. Install it via `pip install inferedge-moss`."
                )

        class DocumentInfo(_MissingDependency):  # type: ignore[no-redef]
            pass

        class IndexInfo(_MissingDependency):  # type: ignore[no-redef]
            pass

        class SearchResult(_MissingDependency):  # type: ignore[no-redef]
            pass

        class AddDocumentsOptions(_MissingDependency):  # type: ignore[no-redef]
            pass

        class GetDocumentsOptions(_MissingDependency):  # type: ignore[no-redef]
            pass

        class QueryOptions(_MissingDependency):  # type: ignore[no-redef]
            pass


from .log import logger

__all__ = [
    "AddDocumentsOptions",
    "DocumentInfo",
    "GetDocumentsOptions",
    "IndexInfo",
    "MossClient",
    "QueryOptions",
    "SearchResult",
]


class MossClient:
    """Async helper around :mod:`inferedge_moss` tailored for LiveKit agents."""

    def __init__(
        self,
        project_id: str | None = None,
        project_key: str | None = None,
    ) -> None:
        if Client is None:
            raise RuntimeError(
                "inferedge-moss is required to use MossClient. Install it via `pip install inferedge-moss`."
            )

        project_id_value = project_id or os.environ.get("MOSS_PROJECT_ID")
        project_key_value = project_key or os.environ.get("MOSS_PROJECT_KEY")

        if not project_id_value:
            raise ValueError(
                "project_id must be provided or set through the MOSS_PROJECT_ID environment variable"
            )

        if not project_key_value:
            raise ValueError(
                "project_key must be provided or set through the MOSS_PROJECT_KEY environment variable"
            )

        self._project_id: str = project_id_value
        self._project_key: str = project_key_value
        self._client = Client(self._project_id, self._project_key)

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

    # ---------- Index lifecycle ----------

    async def create_index(
        self, index_name: str, documents: list[DocumentInfo], model_id: str | None = None
    ) -> bool:
        """Create a new index and populate it with documents."""

        logger.debug("creating moss index", extra={"index": index_name, "model": model_id})
        result = await self._client.create_index(index_name, documents, model_id)
        if not isinstance(result, bool):
            raise TypeError("inferedge_moss.create_index returned an unexpected type")
        return result

    async def get_index(self, index_name: str) -> IndexInfo:
        """Get information about a specific index."""

        logger.debug("getting moss index", extra={"index": index_name})
        index_info = await self._client.get_index(index_name)
        if not isinstance(index_info, IndexInfo):
            raise TypeError("inferedge_moss.get_index returned an unexpected type")
        return index_info

    async def list_indexes(self) -> list[IndexInfo]:
        """List all indexes with their information."""

        logger.debug("listing moss indexes")
        indexes = await self._client.list_indexes()
        if not isinstance(indexes, list):
            raise TypeError("inferedge_moss.list_indexes returned an unexpected type")
        return indexes

    async def delete_index(self, index_name: str) -> bool:
        """Delete an index and all its data."""

        logger.debug("deleting moss index", extra={"index": index_name})
        deleted = await self._client.delete_index(index_name)
        if not isinstance(deleted, bool):
            raise TypeError("inferedge_moss.delete_index returned an unexpected type")
        return bool(deleted)

    # ---------- Document mutations ----------
    async def add_documents(
        self, index_name: str, docs: list[DocumentInfo], options: AddDocumentsOptions | None
    ) -> dict[str, int]:
        """Add or update documents in an index."""

        logger.debug(
            "adding documents to moss index", extra={"index": index_name, "count": len(docs)}
        )
        mapping = await self._client.add_docs(index_name, docs, options)
        if not isinstance(mapping, dict):
            raise TypeError("inferedge_moss.add_docs returned an unexpected type")
        return mapping

    async def delete_docs(self, index_name: str, doc_ids: list[str]) -> dict[str, int]:
        """Delete documents from an index by their IDs."""

        logger.debug(
            "deleting documents from moss index", extra={"index": index_name, "count": len(doc_ids)}
        )
        mapping = await self._client.delete_docs(index_name, doc_ids)
        if not isinstance(mapping, dict):
            raise TypeError("inferedge_moss.delete_docs returned an unexpected type")
        return mapping

    # ---------- View existing documents ----------

    async def get_docs(
        self, index_name: str, options: GetDocumentsOptions | None
    ) -> list[DocumentInfo]:
        """Retrieve documents from an index."""

        logger.debug("retrieving documents from moss index", extra={"index": index_name})
        documents = await self._client.get_docs(index_name, options)
        if not isinstance(documents, list):
            raise TypeError("inferedge_moss.get_docs returned an unexpected type")
        return documents

    # ---------- Index loading & querying ----------

    async def load_index(
        self, index_name: str, auto_refresh: bool = False, polling_interval_in_seconds: int = 600
    ) -> str:
        """Load an index from a local .moss file into memory.

        Args:
            index_name: Name of the index to load
            auto_refresh: Whether to automatically refresh the index from remote
            polling_interval_in_seconds: Interval in seconds between auto-refresh polls
        """

        logger.debug("loading moss index", extra={"index": index_name})
        result = await self._client.load_index(  # type: ignore[call-arg]
            index_name, auto_refresh, polling_interval_in_seconds
        )
        if not isinstance(result, str):
            raise TypeError("moss.load_index returned an unexpected type")
        return result


    async def query(
        self,
        index_name: str,
        query: str,
        top_k: int = 5,
        *,
        options: QueryOptions | None = None,
    ) -> SearchResult:
        """Perform a semantic similarity search against the specified index.

        Args:
            index_name: Name of the index to query
            query: Search query text
            top_k: Number of results to return (default: 5). Ignored if options.top_k is set.
            options: Query options for custom embeddings and advanced settings
        """
        if options is None:
            options = QueryOptions(top_k=top_k)
        elif options.top_k is None:
            options = QueryOptions(top_k=top_k, alpha=options.alpha, embedding=options.embedding)

        logger.debug("querying moss index", extra={"index": index_name, "query": query})

        search_result = await self._client.query(index_name, query, options=options)

        if not isinstance(search_result, SearchResult):
            raise TypeError("moss.query returned an unexpected type")
        return search_result

    def __repr__(self) -> str:
        return f"MossClient(project_id={self._project_id!r})"
