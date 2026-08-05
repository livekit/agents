from __future__ import annotations

import asyncio
from typing import Any

import httpx
import openai
import pytest

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins.openai import stt

pytestmark = pytest.mark.unit


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.fail_next = False

    async def send_json(self, data: dict[str, Any]) -> None:
        if self.fail_next:
            self.fail_next = False
            raise ConnectionResetError("socket went away")
        self.sent.append(data)

    async def receive(self) -> object:
        await asyncio.Event().wait()  # a connection that never says anything
        raise AssertionError("unreachable")

    async def close(self) -> None:
        pass


class _FakeSession:
    def __init__(self) -> None:
        self.ws = _FakeWebSocket()

    async def ws_connect(self, url: str, **_kwargs: object) -> _FakeWebSocket:
        return self.ws


def _offline_stt(**kwargs: Any) -> stt.STT:
    """An STT whose streams connect to a fake socket instead of the network."""
    instance = stt.STT(api_key="test-key", **kwargs)
    instance._session = _FakeSession()  # type: ignore[assignment]
    return instance


def _audio_input(instance: stt.STT) -> dict[str, Any]:
    payload = stt._session_update(instance._opts).model_dump(by_alias=True, exclude_unset=True)
    audio_input = payload["session"]["audio"]["input"]
    assert isinstance(audio_input, dict)
    return audio_input


async def _transcription_config(instance: stt.STT) -> dict[str, Any]:
    config = _audio_input(instance)["transcription"]
    assert isinstance(config, dict)
    return config


async def test_realtime_sends_keywords_and_languages() -> None:
    instance = stt.STT(
        api_key="test-key",
        model="gpt-live-transcribe",
        prompt="A customer support call.",
        keywords=["premium plan", "AC-42"],
        language=["en", "fr"],
    )

    config = await _transcription_config(instance)

    assert config["prompt"] == "A customer support call."
    assert config["keywords"] == ["premium plan", "AC-42"]
    assert config["languages"] == ["en", "fr"]
    # the singular field must not go alongside the plural one
    assert "language" not in config


async def test_realtime_language_codes_are_normalized() -> None:
    # the API only takes ISO-639-1, and rejects regional tags such as en-US
    instance = stt.STT(
        api_key="test-key", model="gpt-transcribe", language=["en-US", "yue", "zh-cn", "zh-tw"]
    )

    config = await _transcription_config(instance)

    # yue survives as its own language; the zh regions collapse into one entry
    assert config["languages"] == ["en", "yue", "zh"]


async def test_realtime_earlier_model_keeps_singular_language() -> None:
    instance = stt.STT(api_key="test-key", model="gpt-4o-mini-transcribe", language="en-US")

    config = await _transcription_config(instance)

    assert config["language"] == "en"
    assert "languages" not in config
    assert "keywords" not in config


async def test_realtime_defaults_to_english() -> None:
    instance = stt.STT(api_key="test-key", model="gpt-4o-mini-transcribe")

    config = await _transcription_config(instance)

    assert config["language"] == "en"


async def test_detect_language_omits_the_language_hint() -> None:
    instance = stt.STT(api_key="test-key", model="gpt-4o-mini-transcribe", detect_language=True)

    config = await _transcription_config(instance)

    assert "language" not in config


async def test_single_language_list_works_on_every_model() -> None:
    instance = stt.STT(api_key="test-key", model="whisper-1", language=["fr"])

    config = await _transcription_config(instance)

    assert config["language"] == "fr"


async def test_multiple_languages_rejected_by_earlier_models() -> None:
    with pytest.raises(ValueError, match="accepts a single language"):
        stt.STT(api_key="test-key", model="gpt-4o-transcribe", language=["en", "fr"])

    instance = stt.STT(api_key="test-key", model="gpt-4o-transcribe")
    with pytest.raises(ValueError, match="accepts a single language"):
        instance.update_options(language=["en", "fr"])
    assert instance._opts.languages == ["en"]


async def test_keywords_rejected_by_earlier_models() -> None:
    with pytest.raises(ValueError, match="keywords are only supported by"):
        stt.STT(api_key="test-key", model="gpt-4o-transcribe", keywords=["AC-42"])


async def test_switching_to_an_earlier_model_rejects_the_current_hints() -> None:
    instance = stt.STT(
        api_key="test-key", model="gpt-live-transcribe", keywords=["AC-42"], language=["en", "fr"]
    )

    with pytest.raises(ValueError, match="keywords are only supported by"):
        instance.update_options(model="gpt-4o-transcribe")

    # nothing was applied
    assert instance._opts.model == "gpt-live-transcribe"
    assert instance.capabilities.keyterms is True
    assert instance._opts.languages == ["en", "fr"]


@pytest.mark.parametrize(
    ("model", "supported"),
    [
        ("gpt-transcribe", True),
        ("gpt-live-transcribe", True),
        ("gpt-4o-transcribe", False),
        ("whisper-1", False),
    ],
)
def test_keyterms_capability_follows_the_model(model: str, supported: bool) -> None:
    instance = stt.STT(api_key="test-key", model=model)
    assert instance.capabilities.keyterms is supported

    instance.update_options(model=model)
    assert instance.capabilities.keyterms is supported


async def _connected(stream: stt.SpeechStream) -> _FakeWebSocket:
    for _ in range(200):
        if stream._ws is not None:
            return stream._ws  # type: ignore[return-value]
        await asyncio.sleep(0.005)
    raise AssertionError("the stream never opened a connection")


async def _settle(stream: stt.SpeechStream) -> None:
    if stream._update_task is not None:
        await asyncio.wait([stream._update_task])


def _sent_transcription(ws: _FakeWebSocket) -> dict[str, Any]:
    config = ws.sent[-1]["session"]["audio"]["input"]["transcription"]
    assert isinstance(config, dict)
    return config


async def test_session_keyterms_merge_behind_user_keywords() -> None:
    instance = _offline_stt(model="gpt-live-transcribe", keywords=["AC-42", "billing"])
    stream = instance.stream()
    ws = await _connected(stream)

    instance._update_session_keyterms(["billing", "Acme Corp"])
    await _settle(stream)

    # user keywords first, no duplicates, and the live connection carries the merge
    assert instance._opts.keywords == ["AC-42", "billing", "Acme Corp"]
    assert _sent_transcription(ws)["keywords"] == ["AC-42", "billing", "Acme Corp"]

    # user keywords survive a keyterm update
    instance._update_session_keyterms([])
    await _settle(stream)
    assert _sent_transcription(ws)["keywords"] == ["AC-42", "billing"]

    await stream.aclose()


async def test_options_apply_without_a_reconnect() -> None:
    instance = _offline_stt(model="gpt-live-transcribe")
    stream = instance.stream()
    ws = await _connected(stream)

    instance.update_options(keywords=["Acme Corp"], prompt="a support call")
    await _settle(stream)
    assert _sent_transcription(ws)["keywords"] == ["Acme Corp"]
    assert _sent_transcription(ws)["prompt"] == "a support call"

    instance.update_options(language="fr")
    await _settle(stream)
    assert _sent_transcription(ws)["languages"] == ["fr"]
    assert stream._language == "fr"

    assert not stream._reconnect_event.is_set()

    await stream.aclose()


async def test_a_failed_update_does_not_break_the_next_one() -> None:
    instance = _offline_stt(model="gpt-live-transcribe")
    stream = instance.stream()
    ws = await _connected(stream)

    ws.fail_next = True
    instance.update_options(keywords=["dropped"])
    await _settle(stream)

    instance.update_options(keywords=["applied"])
    await _settle(stream)
    assert _sent_transcription(ws)["keywords"] == ["applied"]

    await stream.aclose()


async def test_a_new_model_reconnects() -> None:
    # gateways route on the ?model= in the upgrade URL, so this one can't be a session.update
    instance = _offline_stt(model="gpt-live-transcribe")
    stream = instance.stream()
    await _connected(stream)

    instance.update_options(model="gpt-4o-mini-transcribe", language="fr")

    assert stream._reconnect_event.is_set()
    # the transcripts of the rebuilt connection carry the new language, not the old one
    assert stream._language == "fr"

    await stream.aclose()


async def test_cleared_keywords_and_prompt_are_sent_as_empty() -> None:
    # an omitted field keeps its previous value, so clearing has to be explicit
    instance = _offline_stt(model="gpt-live-transcribe", keywords=["AC-42"], prompt="a call")
    stream = instance.stream()
    ws = await _connected(stream)

    instance.update_options(keywords=[], prompt="")
    await _settle(stream)

    assert _sent_transcription(ws)["keywords"] == []
    assert _sent_transcription(ws)["prompt"] == ""

    await stream.aclose()


async def test_clearing_the_language_reconnects() -> None:
    # the API rejects both an empty array and null for `languages`
    instance = _offline_stt(model="gpt-live-transcribe", language=["en", "fr"])
    stream = instance.stream()
    await _connected(stream)

    instance.update_options(detect_language=True)

    assert instance._opts.languages == []
    assert stream._reconnect_event.is_set()

    await stream.aclose()


async def test_switching_to_a_client_commit_model_needs_a_vad() -> None:
    instance = stt.STT(api_key="test-key", model="gpt-4o-mini-transcribe", use_realtime=True)

    with pytest.raises(ValueError, match="no server-side endpointing"):
        instance.update_options(model="gpt-live-transcribe")

    # an explicit vad=None means the caller commits the buffer itself
    opted_out = stt.STT(
        api_key="test-key", model="gpt-4o-mini-transcribe", use_realtime=True, vad=None
    )
    opted_out.update_options(model="gpt-live-transcribe")
    assert opted_out._opts.model == "gpt-live-transcribe"


async def test_detected_keyterms_do_not_block_a_new_stream() -> None:
    instance = _offline_stt(model="gpt-live-transcribe")
    instance._update_session_keyterms(["Acme Corp"])
    instance.update_options(model="gpt-4o-mini-transcribe", keywords=[])

    # the detected terms are gone, so an explicit language does not trip validation
    stream = instance.stream(language="fr")
    assert instance._opts.keywords == []

    await stream.aclose()


async def test_detected_keyterms_do_not_follow_a_model_downgrade() -> None:
    instance = _offline_stt(model="gpt-live-transcribe")
    instance._update_session_keyterms(["Acme Corp"])

    # clearing the user keywords passes validation, but the detected ones must not leak
    instance.update_options(model="gpt-4o-mini-transcribe", keywords=[])

    config = await _transcription_config(instance)
    assert "keywords" not in config


async def test_live_transcribe_omits_turn_detection() -> None:
    # the model rejects any turn_detection config and needs a client-side commit
    instance = stt.STT(api_key="test-key", model="gpt-live-transcribe", use_realtime=True, vad=None)

    assert "turn_detection" not in _audio_input(instance)

    # every other model still gets server-side VAD
    other = stt.STT(api_key="test-key", model="gpt-4o-mini-transcribe", use_realtime=True)
    assert _audio_input(other)["turn_detection"]["type"] == "server_vad"


async def test_a_reused_connection_is_reconfigured() -> None:
    # the pool hands back sockets it configured earlier, so acquiring must re-send the config
    instance = _offline_stt(model="gpt-live-transcribe")
    first = instance.stream()
    ws = await _connected(first)
    await first.aclose()

    # the change lands while nothing is streaming, so no live socket receives it
    instance.update_options(keywords=["Acme Corp"])
    assert _sent_transcription(ws)["keywords"] == []  # the idle socket has not seen it

    second = instance.stream()
    await _connected(second)

    assert _sent_transcription(ws)["keywords"] == ["Acme Corp"]

    await second.aclose()


async def test_session_keyterms_reach_the_next_connection() -> None:
    instance = stt.STT(api_key="test-key", model="gpt-live-transcribe")
    instance._update_session_keyterms(["Acme Corp"])

    config = await _transcription_config(instance)

    assert config["keywords"] == ["Acme Corp"]


async def _transcription_form(instance: stt.STT, captured: list[httpx.Request]) -> str:
    frame = rtc.AudioFrame(
        data=b"\x00\x00" * 2400, sample_rate=24000, num_channels=1, samples_per_channel=2400
    )
    event = await instance._recognize_impl(frame, conn_options=DEFAULT_API_CONNECT_OPTIONS)
    assert event.alternatives[0].text == "hello"
    return captured[0].read().decode("utf-8", errors="replace")


def _mock_client(captured: list[httpx.Request]) -> openai.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"text": "hello"})

    return openai.AsyncClient(
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def test_file_transcription_sends_bracketed_form_fields() -> None:
    captured: list[httpx.Request] = []
    instance = stt.STT(
        model="gpt-transcribe",
        client=_mock_client(captured),
        keywords=["premium plan", "AC-42"],
        language=["en", "fr"],
    )

    body = await _transcription_form(instance, captured)

    assert 'name="keywords[]"\r\n\r\npremium plan' in body
    assert 'name="keywords[]"\r\n\r\nAC-42' in body
    assert 'name="languages[]"\r\n\r\nen' in body
    assert 'name="languages[]"\r\n\r\nfr' in body
    assert 'name="language"' not in body


async def test_file_transcription_earlier_model_keeps_singular_language() -> None:
    captured: list[httpx.Request] = []
    instance = stt.STT(model="whisper-1", client=_mock_client(captured), language="en-US")

    body = await _transcription_form(instance, captured)

    assert 'name="language"\r\n\r\nen' in body
    assert 'name="languages[]"' not in body
