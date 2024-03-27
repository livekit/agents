from __future__ import annotations

import base64
import hashlib
import secrets
import string
from base64 import urlsafe_b64decode
from json import loads
from typing import Any, Dict, Type, TypeVar, Union, cast

from httpx import HTTPStatusError
from pydantic import BaseModel

from .errors import AuthApiError, AuthError, AuthRetryableError, AuthUnknownError
from .types import (
    AuthOtpResponse,
    AuthResponse,
    GenerateLinkProperties,
    GenerateLinkResponse,
    Session,
    SSOResponse,
    User,
    UserResponse,
)

TBaseModel = TypeVar("TBaseModel", bound=BaseModel)


def model_validate(model: Type[TBaseModel], contents) -> TBaseModel:
    """Compatibility layer between pydantic 1 and 2 for parsing an instance
    of a BaseModel from varied"""
    try:
        # pydantic > 2
        return model.model_validate(contents)
    except AttributeError:
        # pydantic < 2
        return model.parse_obj(contents)


def model_dump(model: BaseModel) -> Dict[str, Any]:
    """Compatibility layer between pydantic 1 and 2 for dumping a model's contents as a dict"""
    try:
        # pydantic > 2
        return model.model_dump()
    except AttributeError:
        # pydantic < 2
        return model.dict()


def model_dump_json(model: BaseModel) -> str:
    """Compatibility layer between pydantic 1 and 2 for dumping a model's contents as json"""
    try:
        # pydantic > 2
        return model.model_dump_json()
    except AttributeError:
        # pydantic < 2
        return model.json()


def parse_auth_response(data: Any) -> AuthResponse:
    session: Union[Session, None] = None
    if (
        "access_token" in data
        and "refresh_token" in data
        and "expires_in" in data
        and data["access_token"]
        and data["refresh_token"]
        and data["expires_in"]
    ):
        session = model_validate(Session, data)
    user_data = data.get("user", data)
    user = model_validate(User, user_data) if user_data else None
    return AuthResponse(session=session, user=user)


def parse_auth_otp_response(data: Any) -> AuthOtpResponse:
    return model_validate(AuthOtpResponse, data)


def parse_link_response(data: Any) -> GenerateLinkResponse:
    properties = GenerateLinkProperties(
        action_link=data.get("action_link"),
        email_otp=data.get("email_otp"),
        hashed_token=data.get("hashed_token"),
        redirect_to=data.get("redirect_to"),
        verification_type=data.get("verification_type"),
    )
    user = model_validate(
        User, {k: v for k, v in data.items() if k not in model_dump(properties)}
    )
    return GenerateLinkResponse(properties=properties, user=user)


def parse_user_response(data: Any) -> UserResponse:
    if "user" not in data:
        data = {"user": data}
    return model_validate(UserResponse, data)


def parse_sso_response(data: Any) -> SSOResponse:
    return model_validate(SSOResponse, data)


def get_error_message(error: Any) -> str:
    props = ["msg", "message", "error_description", "error"]
    filter = (
        lambda prop: prop in error if isinstance(error, dict) else hasattr(error, prop)
    )
    return next((error[prop] for prop in props if filter(prop)), str(error))


def looks_like_http_status_error(exception: Exception) -> bool:
    return isinstance(exception, HTTPStatusError)


def handle_exception(exception: Exception) -> AuthError:
    if not looks_like_http_status_error(exception):
        return AuthRetryableError(get_error_message(exception), 0)
    error = cast(HTTPStatusError, exception)
    try:
        network_error_codes = [502, 503, 504]
        if error.response.status_code in network_error_codes:
            return AuthRetryableError(
                get_error_message(error), error.response.status_code
            )
        json = error.response.json()
        return AuthApiError(get_error_message(json), error.response.status_code or 500)
    except Exception as e:
        return AuthUnknownError(get_error_message(error), e)


def decode_jwt_payload(token: str) -> Any:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("JWT is not valid: not a JWT structure")
    base64Url = parts[1]
    # Addding padding otherwise the following error happens:
    # binascii.Error: Incorrect padding
    base64UrlWithPadding = base64Url + "=" * (-len(base64Url) % 4)
    return loads(urlsafe_b64decode(base64UrlWithPadding).decode("utf-8"))


def generate_pkce_verifier(length=64):
    """Generate a random PKCE verifier of the specified length."""
    if length < 43 or length > 128:
        raise ValueError("PKCE verifier length must be between 43 and 128 characters")

    # Define characters that can be used in the PKCE verifier
    charset = string.ascii_letters + string.digits + "-._~"

    return "".join(secrets.choice(charset) for _ in range(length))


def generate_pkce_challenge(code_verifier):
    """Generate a code challenge from a PKCE verifier."""
    # Hash the verifier using SHA-256
    verifier_bytes = code_verifier.encode("utf-8")
    sha256_hash = hashlib.sha256(verifier_bytes).digest()

    return base64.urlsafe_b64encode(sha256_hash).rstrip(b"=").decode("utf-8")
