from __future__ import annotations

import pytest

from livekit.plugins.google.beta.gemini_tts import TTS as GeminiTTS
from livekit.plugins.google.llm import LLM as GoogleLLM

pytestmark = pytest.mark.unit


async def test_llm_close_releases_the_genai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LLM.aclose()`` must release the genai http clients.

    Sibling of ``test_session_close_releases_the_genai_client``, which covers the
    realtime session. Otherwise the clients live until the collector runs
    ``AsyncClient.__del__``, which does
    ``asyncio.get_running_loop().create_task(self.aclose())`` - creating pending
    tasks on whatever event loop is running at that moment.

    ``llm.LLM`` implements ``__aexit__ -> aclose()``, so ``async with`` is the
    documented lifecycle and is what this drives.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    llm = GoogleLLM(model="gemini-2.0-flash-001")

    closed = False
    real_aclose = llm._client.aio.aclose

    async def _spy() -> None:
        nonlocal closed
        closed = True
        await real_aclose()

    monkeypatch.setattr(llm._client.aio, "aclose", _spy)

    async with llm:
        pass

    assert closed
    assert llm._client._api_client._async_httpx_client.is_closed


async def test_gemini_tts_close_releases_the_genai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``TTS.aclose()`` must release the genai http clients.

    ``tts.TTS.aclose`` is an empty default rather than abstract, so a plugin that
    owns a client and does not override it leaks silently.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    tts = GeminiTTS()

    closed = False
    real_aclose = tts._client.aio.aclose

    async def _spy() -> None:
        nonlocal closed
        closed = True
        await real_aclose()

    monkeypatch.setattr(tts._client.aio, "aclose", _spy)

    async with tts:
        pass

    assert closed
    assert tts._client._api_client._async_httpx_client.is_closed


async def test_llm_close_survives_a_failing_client_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genai client that raises on close must not break ``aclose()``.

    ``aclose()`` runs on the shutdown path, so a provider-side failure here has to
    stay contained - the same reason the realtime fix wraps its release in
    ``try/except``.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    llm = GoogleLLM(model="gemini-2.0-flash-001")

    async def _boom() -> None:
        raise RuntimeError("client close failed")

    monkeypatch.setattr(llm._client.aio, "aclose", _boom)

    # must not propagate
    async with llm:
        pass
