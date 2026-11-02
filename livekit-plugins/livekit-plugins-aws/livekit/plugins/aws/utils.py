from __future__ import annotations

from aiobotocore.session import AioSession, get_session  # type: ignore

from .log import logger

DEFAULT_REGION = "us-east-1"


def _strip_nones(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _resolve_session(session: object | None) -> AioSession:
    """Return an ``AioSession`` for the given ``session`` argument.

    Accepts ``None`` (creates a fresh session), an ``aiobotocore.session.AioSession``,
    or — for backwards compatibility — a legacy ``aioboto3.Session``, which wraps an
    ``AioSession`` and is unwrapped here with a ``DeprecationWarning``. ``aioboto3`` is
    imported lazily so it does not become a dependency of this plugin.
    """
    if session is None:
        return get_session()
    if isinstance(session, AioSession):
        return session

    try:
        import aioboto3  # type: ignore
    except ImportError:
        aioboto3 = None

    if aioboto3 is not None and isinstance(session, aioboto3.Session):
        logger.warning(
            "Passing an aioboto3.Session is deprecated; pass an "
            "aiobotocore.session.AioSession instead. The AWS plugin no longer depends "
            "on aioboto3."
        )
        return session._session

    raise TypeError(
        "session must be an aiobotocore.session.AioSession "
        f"(or a legacy aioboto3.Session), got {type(session).__name__}."
    )
