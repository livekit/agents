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
    """Retryable: the connection was lost, the database is unreachable or busy, or it answered
    something that did not decode."""


class LeaseLost(StorageError):
    """The run's lane lease was taken, released or could not be renewed in time."""


class RunRejected(StorageError):
    """The lane refused the run: the reject policy refused the message while a turn runs."""

    running_task_id: str | None
    """The running turn's task, when the journal names it."""

    def __init__(self, message: str, *, running_task_id: str | None = None) -> None:
        super().__init__(message)
        self.running_task_id = running_task_id


__all__ = [
    "LeaseLost",
    "NotSerializable",
    "RunRejected",
    "StorageError",
    "StorageUnavailable",
    "ThreadNotFound",
    "TooLarge",
    "VersionConflict",
]
