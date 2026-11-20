from __future__ import annotations

from typing import Any

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, stt
from livekit.plugins.clova import STT

pytestmark = pytest.mark.unit

_NO_RETRY = APIConnectOptions(max_retry=0, retry_interval=0, timeout=1)


class _FakeResponse:
    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self._body = body

    async def json(self, *args: Any, **kwargs: Any) -> Any:
        return self._body

    async def text(self) -> str:
        return str(self._body)

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeSession:
    def __init__(self, status: int, body: Any) -> None:
        self._response = _FakeResponse(status, body)
        self.calls = 0

    def post(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        self.calls += 1
        return self._response


def _make_stt(status: int, body: Any) -> tuple[STT, _FakeSession]:
    session = _FakeSession(status, body)
    clova = STT(
        secret="secret",
        invoke_url="https://clova.invalid",
        http_session=session,  # type: ignore[arg-type]
    )
    return clova, session


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * 1600, sample_rate=16000, num_channels=1, samples_per_channel=1600
    )


async def test_recognize_returns_final_transcript() -> None:
    clova, _ = _make_stt(200, {"text": "hello world", "confidence": 0.9})

    event = await clova.recognize(_frame(), conn_options=_NO_RETRY)

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "hello world"


@pytest.mark.parametrize(
    "body",
    [
        {"text": "", "confidence": 0.0},
        {"text": None},
        {},
    ],
    ids=["empty-text", "null-text", "no-text"],
)
async def test_silence_returns_empty_final_transcript(body: dict[str, Any]) -> None:
    clova, _ = _make_stt(200, body)

    event = await clova.recognize(_frame(), conn_options=_NO_RETRY)

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == ""


async def test_low_confidence_returns_empty_final_transcript() -> None:
    clova, _ = _make_stt(200, {"text": "cough", "confidence": 0.1})

    event = await clova.recognize(_frame(), conn_options=_NO_RETRY)

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == ""


async def test_missing_confidence_is_accepted() -> None:
    clova, _ = _make_stt(200, {"text": "hello"})

    event = await clova.recognize(_frame(), conn_options=_NO_RETRY)

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "hello"


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(401, False), (503, True)],
)
async def test_http_error_raises_api_status_error(status: int, retryable: bool) -> None:
    clova, _ = _make_stt(status, {"message": "boom"})

    with pytest.raises(APIStatusError) as exc_info:
        await clova.recognize(_frame(), conn_options=_NO_RETRY)

    assert exc_info.value.status_code == status
    assert exc_info.value.retryable is retryable


async def test_error_in_body_is_not_treated_as_silence() -> None:
    clova, _ = _make_stt(200, {"error": "bad request"})

    with pytest.raises(APIStatusError):
        await clova.recognize(_frame(), conn_options=_NO_RETRY)


async def test_client_response_error_is_wrapped() -> None:
    clova, _ = _make_stt(200, {})

    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise aiohttp.ClientResponseError(
            request_info=None,  # type: ignore[arg-type]
            history=(),
            status=502,
            message="bad gateway",
        )

    clova._session.post = _raise  # type: ignore[union-attr, method-assign]

    with pytest.raises(APIStatusError) as exc_info:
        await clova.recognize(_frame(), conn_options=_NO_RETRY)

    assert exc_info.value.status_code == 502
