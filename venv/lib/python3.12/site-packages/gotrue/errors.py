from __future__ import annotations

from typing import Union

from typing_extensions import TypedDict


class AuthError(Exception):
    def __init__(self, message: str) -> None:
        Exception.__init__(self, message)
        self.message = message
        self.name = "AuthError"


class AuthApiErrorDict(TypedDict):
    name: str
    message: str
    status: int


class AuthApiError(AuthError):
    def __init__(self, message: str, status: int) -> None:
        AuthError.__init__(self, message)
        self.name = "AuthApiError"
        self.status = status

    def to_dict(self) -> AuthApiErrorDict:
        return {
            "name": self.name,
            "message": self.message,
            "status": self.status,
        }


class AuthUnknownError(AuthError):
    def __init__(self, message: str, original_error: Exception) -> None:
        AuthError.__init__(self, message)
        self.name = "AuthUnknownError"
        self.original_error = original_error


class CustomAuthError(AuthError):
    def __init__(self, message: str, name: str, status: int) -> None:
        AuthError.__init__(self, message)
        self.name = name
        self.status = status

    def to_dict(self) -> AuthApiErrorDict:
        return {
            "name": self.name,
            "message": self.message,
            "status": self.status,
        }


class AuthSessionMissingError(CustomAuthError):
    def __init__(self) -> None:
        CustomAuthError.__init__(
            self,
            "Auth session missing!",
            "AuthSessionMissingError",
            400,
        )


class AuthInvalidCredentialsError(CustomAuthError):
    def __init__(self, message: str) -> None:
        CustomAuthError.__init__(
            self,
            message,
            "AuthInvalidCredentialsError",
            400,
        )


class AuthImplicitGrantRedirectErrorDetails(TypedDict):
    error: str
    code: str


class AuthImplicitGrantRedirectErrorDict(AuthApiErrorDict):
    details: Union[AuthImplicitGrantRedirectErrorDetails, None]


class AuthImplicitGrantRedirectError(CustomAuthError):
    def __init__(
        self,
        message: str,
        details: Union[AuthImplicitGrantRedirectErrorDetails, None] = None,
    ) -> None:
        CustomAuthError.__init__(
            self,
            message,
            "AuthImplicitGrantRedirectError",
            500,
        )
        self.details = details

    def to_dict(self) -> AuthImplicitGrantRedirectErrorDict:
        return {
            "name": self.name,
            "message": self.message,
            "status": self.status,
            "details": self.details,
        }


class AuthRetryableError(CustomAuthError):
    def __init__(self, message: str, status: int) -> None:
        CustomAuthError.__init__(
            self,
            message,
            "AuthRetryableError",
            status,
        )
