from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from livekit.agents import LanguageCode

pytestmark = pytest.mark.unit

with pytest.warns(DeprecationWarning):
    from livekit.plugins.turn_detector import base, multilingual


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.requests: list[dict[str, Any]] = []

    def post(self, *, url: str, json: dict[str, Any], **kwargs: Any) -> _FakeResponse:
        self.requests.append(json)
        return _FakeResponse(self._payload)


@pytest.fixture
def remote_session(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    session = _FakeSession({"threshold": 0.0123})
    monkeypatch.setenv("LIVEKIT_REMOTE_EOT_URL", "http://eot.invalid")
    monkeypatch.setattr(
        base, "get_job_context", lambda: SimpleNamespace(inference_executor=object())
    )
    monkeypatch.setattr(multilingual.utils.http_context, "http_session", lambda: session)
    return session


async def test_remote_threshold_used_without_override(remote_session: _FakeSession) -> None:
    model = multilingual.MultilingualModel()

    assert await model.unlikely_threshold(LanguageCode("es")) == 0.0123
    assert await model.unlikely_threshold(LanguageCode("es")) == 0.0123
    # the second lookup is served from the cache
    assert len(remote_session.requests) == 1


async def test_custom_threshold_overrides_remote_threshold(remote_session: _FakeSession) -> None:
    model = multilingual.MultilingualModel(unlikely_threshold=0.5)

    # the first lookup for a language fetches the server threshold; the custom one
    # must still win, as it does for every later lookup served from the cache
    assert await model.unlikely_threshold(LanguageCode("es")) == 0.5
    assert await model.unlikely_threshold(LanguageCode("es")) == 0.5
    assert len(remote_session.requests) == 1


async def test_custom_threshold_for_unsupported_remote_language(
    remote_session: _FakeSession,
) -> None:
    remote_session._payload = {}
    model = multilingual.MultilingualModel(unlikely_threshold=0.5)

    # the server has no threshold for the language, so it stays unsupported
    assert await model.unlikely_threshold(LanguageCode("xx")) is None
    assert not await model.supports_language(LanguageCode("xx"))
