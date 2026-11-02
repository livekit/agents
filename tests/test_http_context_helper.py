"""Tests for the public `utils.http_context.open()` helper."""

from __future__ import annotations

import asyncio
import os
import ssl
from pathlib import Path

import aiohttp
import certifi
import pytest

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


@pytest.fixture(autouse=True)
def _fresh_ssl_context_cache() -> None:
    """Each test builds its context under its own env and verify paths."""
    http_context._ssl_context_for.cache_clear()


def test_ssl_context_is_built_once_per_trust_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every job's session reuses one context instead of parsing the CA bundle again."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    certifi_count = _certifi_cert_count()
    built: list[ssl.SSLContext] = []
    real_create = ssl.create_default_context

    def _counting(*args: object, **kwargs: object) -> ssl.SSLContext:
        built.append(real_create(*args, **kwargs))  # type: ignore[arg-type]
        return built[-1]

    monkeypatch.setattr(ssl, "create_default_context", _counting)

    system_ctx = http_context._create_ssl_context()
    assert http_context._create_ssl_context() is system_ctx
    assert len(built) == 1

    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    certifi_ctx = http_context._create_ssl_context()
    assert certifi_ctx is not system_ctx
    assert certifi_ctx.cert_store_stats()["x509"] == certifi_count
    assert http_context._create_ssl_context() is certifi_ctx
    assert len(built) == 2


def _certifi_cert_count() -> int:
    ctx = ssl.create_default_context(cafile=certifi.where())
    return ctx.cert_store_stats()["x509"]


def test_ssl_context_falls_back_to_certifi_without_system_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the host has no resolvable system trust store, certifi is loaded.

    Asserts on the actual ``load_verify_locations(certifi.where())`` call, not
    cert counts: ``create_default_context()`` loads OpenSSL's compiled-in roots
    at the C level regardless of the ``get_default_verify_paths`` monkeypatch,
    so a count check would pass on a host with a system store even if the
    certifi branch never ran.
    """
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

    loaded_cafiles: list[str | None] = []
    real_load = ssl.SSLContext.load_verify_locations

    def _spy(self: ssl.SSLContext, *args: object, **kwargs: object) -> object:
        loaded_cafiles.append(kwargs.get("cafile"))  # type: ignore[arg-type]
        return real_load(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ssl.SSLContext, "load_verify_locations", _spy)

    http_context._create_ssl_context()
    assert certifi.where() in loaded_cafiles


def test_set_default_cert_env_sets_var_without_system_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no env override and no system store, SSL_CERT_FILE is set to certifi."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    paths = ssl.get_default_verify_paths()._replace(cafile=None, capath=None)
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: paths)

    http_context._set_default_cert_env()
    assert os.environ.get("SSL_CERT_FILE") == certifi.where()


def test_set_default_cert_env_shares_system_paths_with_native_clients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Native clients must use the same resolved file and directory as Python."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    paths = ssl.get_default_verify_paths()._replace(cafile=certifi.where(), capath=str(tmp_path))
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: paths)

    http_context._set_default_cert_env()
    assert os.environ["SSL_CERT_FILE"] == certifi.where()
    assert os.environ["SSL_CERT_DIR"] == str(tmp_path)


def test_set_default_cert_env_respects_existing_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit SSL_CERT_FILE is never overwritten, even with no system store."""
    monkeypatch.setenv("SSL_CERT_FILE", "/custom/bundle.pem")
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setattr(http_context, "_has_system_trust_store", lambda: False)

    http_context._set_default_cert_env()
    assert os.environ["SSL_CERT_FILE"] == "/custom/bundle.pem"
