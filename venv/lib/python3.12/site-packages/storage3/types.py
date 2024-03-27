from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

import dateutil.parser
from typing_extensions import Literal, TypedDict

RequestMethod = Literal["GET", "POST", "DELETE", "PUT", "HEAD"]


@dataclass
class BaseBucket:
    """Represents a file storage bucket."""

    id: str
    name: str
    owner: str
    public: bool
    created_at: datetime
    updated_at: datetime
    file_size_limit: Optional[int]
    allowed_mime_types: Optional[list[str]]

    def __post_init__(self) -> None:
        # created_at and updated_at are returned by the API as ISO timestamps
        # so we convert them to datetime objects
        self.created_at = dateutil.parser.isoparse(self.created_at)  # type: ignore
        self.updated_at = dateutil.parser.isoparse(self.updated_at)  # type: ignore


# used in bucket.list method's option parameter
class _sortByType(TypedDict):
    column: str
    order: Literal["asc", "desc"]


class SignedUploadURL(TypedDict):
    signed_url: str
    token: str
    path: str


class CreateOrUpdateBucketOptions(TypedDict, total=False):
    public: bool
    file_size_limit: int
    allowed_mime_types: list[str]


class ListBucketFilesOptions(TypedDict):
    limit: int
    offset: int
    sortBy: _sortByType


class TransformOptions(TypedDict, total=False):
    height: int
    width: int
    resize: Literal["cover", "contain", "fill"]
    format: Literal["origin", "avif"]
    quality: int


class URLOptions(TypedDict, total=False):
    download: Union[str, bool]
    transform: TransformOptions


class CreateSignedURLsOptions(TypedDict, total=False):
    download: Union[str, bool]


class DownloadOptions(TypedDict, total=False):
    transform: TransformOptions


FileOptions = TypedDict(
    "FileOptions",
    {"cache-control": str, "content-type": str, "x-upsert": str, "upsert": str},
    total=False,
)
