from __future__ import annotations

from aiobotocore.session import AioSession, get_session

DEFAULT_REGION = "us-east-1"


def resolve_session(session: AioSession | None) -> AioSession:
    return session if session is not None else get_session()
