"""Tests for the public `utils.http_context.open()` helper and for the
inference STT error surface when called outside a job context.
"""

from __future__ import annotations

import asyncio
import os
import ssl

import aiohttp
import certifi
import pytest

from livekit.agents import inference
from livekit.agents.utils import http_context

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]


async def test_open_yields_working_session_and_closes_on_exit() -> None:
    with pytest.raises(RuntimeError):
        http_context.http_session()

    async with http_context.open() as session:
        assert isinstance(session, aiohttp.ClientSession)
        assert not session.closed
        # http_session() returns the same instance inside the block
        assert http_context.http_session() is session

    assert session.closed
    with pytest.raises(RuntimeError):
        http_context.http_session()


async def test_open_is_reentrant_inner_does_not_close_outer() -> None:
    async with http_context.open() as outer:
        async with http_context.open() as inner:
            # nested open() reuses the outer session — does not create a new one
            assert inner is outer

        # outer session is untouched after inner exits
        assert not outer.closed
        assert http_context.http_session() is outer

    assert outer.closed


async def test_open_isolated_per_task() -> None:
    """Each asyncio.Task gets its own http session context — they don't share."""
    barrier = asyncio.Event()

    async def worker() -> tuple[aiohttp.ClientSession, bool]:
        async with http_context.open() as session:
            await barrier.wait()
            still_open = not session.closed
        return session, still_open

    task_a = asyncio.create_task(worker())
    task_b = asyncio.create_task(worker())
    await asyncio.sleep(0.01)
    barrier.set()

    (sess_a, a_open), (sess_b, b_open) = await asyncio.gather(task_a, task_b)
    assert sess_a is not sess_b
    assert a_open and b_open
    assert sess_a.closed and sess_b.closed


async def test_http_session_error_message_points_to_helper() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        http_context.http_session()
    msg = str(exc_info.value)
    assert "http_context.open()" in msg


def _certifi_cert_count() -> int:
    ctx = ssl.create_default_context(cafile=certifi.where())
    return ctx.cert_store_stats()["x509"]


def test_ssl_context_honors_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """SSL_CERT_FILE / SSL_CERT_DIR take precedence over the system store."""
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)

    ctx = http_context._create_ssl_context()
    assert ctx.cert_store_stats()["x509"] == _certifi_cert_count()


def test_ssl_context_falls_back_to_certifi_without_system_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the host has no resolvable system trust store, certifi is loaded."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setattr(
        ssl,
        "get_default_verify_paths",
        lambda: ssl.DefaultVerifyPaths(
            cafile="/nonexistent/cert.pem",
            capath="/nonexistent/certs",
            openssl_cafile_env="SSL_CERT_FILE",
            openssl_cafile="/nonexistent/cert.pem",
            openssl_capath_env="SSL_CERT_DIR",
            openssl_capath="/nonexistent/certs",
        ),
    )

    ctx = http_context._create_ssl_context()
    assert ctx.cert_store_stats()["x509"] >= _certifi_cert_count()


def test_ssl_context_uses_system_store_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolvable system store is used as-is, without forcing certifi in."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)

    paths = ssl.get_default_verify_paths()
    has_system_store = bool(
        (paths.cafile and os.path.exists(paths.cafile))
        or (paths.capath and os.path.isdir(paths.capath))
    )
    if not has_system_store:
        pytest.skip("host has no system trust store to exercise this path")

    ctx = http_context._create_ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED


async def test_inference_stt_surfaces_real_error_outside_ctx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: previously, calling `inference.STT().stream()` outside a job
    context raised `AttributeError: 'SpeechStream' object has no attribute
    '_session'` from a background task — masking the real "no http context" error.

    After the fix, _ensure_session() runs inside _run(), so the actual
    RuntimeError surfaces through the stream's main task.
    """
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret")

    stt = inference.STT(model="cartesia/sonic-3")
    stream = stt.stream()

    # SpeechStream no longer eagerly grabs `_session` in __init__.
    assert not hasattr(stream, "_session")

    with pytest.raises(RuntimeError, match="http_context.open"):
        await stream._task

    await stream.aclose()
