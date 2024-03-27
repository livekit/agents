from __future__ import annotations

from typing import Any, Optional

from httpx import HTTPError, Response

from ..types import CreateOrUpdateBucketOptions, RequestMethod
from ..utils import StorageException, SyncClient
from .file_api import SyncBucket

__all__ = ["SyncStorageBucketAPI"]


class SyncStorageBucketAPI:
    """This class abstracts access to the endpoint to the Get, List, Empty, and Delete operations on a bucket"""

    def __init__(self, session: SyncClient) -> None:
        self._client = session

    def _request(
        self,
        method: RequestMethod,
        url: str,
        json: Optional[dict[Any, Any]] = None,
    ) -> Response:
        response = self._client.request(method, url, json=json)
        try:
            response.raise_for_status()
        except HTTPError:
            raise StorageException(
                {**response.json(), "statusCode": response.status_code}
            )

        return response

    def list_buckets(self) -> list[SyncBucket]:
        """Retrieves the details of all storage buckets within an existing product."""
        # if the request doesn't error, it is assured to return a list
        res = self._request("GET", "/bucket")
        return [SyncBucket(**bucket, _client=self._client) for bucket in res.json()]

    def get_bucket(self, id: str) -> SyncBucket:
        """Retrieves the details of an existing storage bucket.

        Parameters
        ----------
        id
            The unique identifier of the bucket you would like to retrieve.
        """
        res = self._request("GET", f"/bucket/{id}")
        json = res.json()
        return SyncBucket(**json, _client=self._client)

    def create_bucket(
        self,
        id: str,
        name: Optional[str] = None,
        options: Optional[CreateOrUpdateBucketOptions] = None,
    ) -> dict[str, str]:
        """Creates a new storage bucket.

        Parameters
        ----------
        id
            A unique identifier for the bucket you are creating.
        name
            A name for the bucket you are creating. If not passed, the id is used as the name as well.
        options
            Extra options to send while creating the bucket. Valid options are `public`, `file_size_limit` and
            `allowed_mime_types`.
        """
        json: dict[str, Any] = {"id": id, "name": name or id}
        if options:
            json.update(**options)
        res = self._request(
            "POST",
            "/bucket",
            json=json,
        )
        return res.json()

    def update_bucket(
        self, id: str, options: CreateOrUpdateBucketOptions
    ) -> dict[str, str]:
        """Update a storage bucket.

        Parameters
        ----------
        id
            The unique identifier of the bucket you would like to update.
        options
            The properties you want to update. Valid options are `public`, `file_size_limit` and
            `allowed_mime_types`.
        """
        json = {"id": id, "name": id, **options}
        res = self._request("PUT", f"/bucket/{id}", json=json)
        return res.json()

    def empty_bucket(self, id: str) -> dict[str, str]:
        """Removes all objects inside a single bucket.

        Parameters
        ----------
        id
            The unique identifier of the bucket you would like to empty.
        """
        res = self._request("POST", f"/bucket/{id}/empty", json={})
        return res.json()

    def delete_bucket(self, id: str) -> dict[str, str]:
        """Deletes an existing bucket. Note that you cannot delete buckets with existing objects inside. You must first
        `empty()` the bucket.

        Parameters
        ----------
        id
            The unique identifier of the bucket you would like to delete.
        """
        res = self._request("DELETE", f"/bucket/{id}", json={})
        return res.json()
