"""The storage errors: what ``lk_agents_abi::storage::StorageError`` raises, and what a door's
user catches."""

from __future__ import annotations


class StorageError(RuntimeError):
    """A session storage failure."""


class ThreadNotFound(StorageError):
    """An unknown, deleted or expired thread."""


class VersionConflict(StorageError):
    """A fenced commit matched no row: the lane moved past the version the run loaded."""


class NotSerializable(StorageError):
    """A field that could not be serialized; the message names it."""


class TooLarge(StorageError):
    """A value over its size limit; the message names what and the limit."""


class StorageUnavailable(StorageError):
    """Retryable: the connection was lost, the lease lapsed, or a transaction was discarded by
    a reconnect."""


class RunRejected(StorageError):
    """The reject policy refused the message while a turn runs."""

    def __init__(self, message: str, *, running_task_id: str | None = None) -> None:
        super().__init__(message)
        self.running_task_id = running_task_id

    """The running turn's task, when the journal names it."""


__all__ = [
    "NotSerializable",
    "RunRejected",
    "StorageError",
    "StorageUnavailable",
    "ThreadNotFound",
    "TooLarge",
    "VersionConflict",
]
