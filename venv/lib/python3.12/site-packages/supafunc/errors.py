from __future__ import annotations

from typing import TypedDict


class FunctionsApiErrorDict(TypedDict):
    name: str
    message: str
    status: int


class FunctionsError(Exception):
    def __init__(self, message: str, name: str, status: int) -> None:
        super().__init__(message)
        self.message = message
        self.name = name
        self.status = status

    def to_dict(self) -> FunctionsApiErrorDict:
        return {
            "name": self.name,
            "message": self.message,
            "status": self.status,
        }


class FunctionsHttpError(FunctionsError):
    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            "FunctionsHttpError",
            400,
        )


class FunctionsRelayError(FunctionsError):
    """Base exception for relay errors."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            "FunctionsRelayError",
            400,
        )
