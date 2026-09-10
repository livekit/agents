from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import aiohttp
import httpx
import openai
import pytest

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, inference, vad
from livekit.agents.stt import SpeechEvent, SpeechEventType
from livekit.plugins.openai import stt

pytestmark = pytest.mark.unit


class _FakeMessage:
    type = aiohttp.WSMsgType.TEXT

    def __init__(self, payload: dict[str, Any]) -> None:
        self.data = json.dumps(payload)


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.fail_next = False
        self.block_first: asyncio.Event | None = None
        self.closed = False

    async def send_json(self, data: dict[str, Any]) -> None:
        if self.fail_next:
            self.fail_next = False
            raise ConnectionResetError("socket went away")
        if self.block_first is not None:
            blocked, self.block_first = self.block_first, None
            await blocked.wait()
        self.sent.append(data)

    async def receive(self) -> object:
        return _FakeMessage(await self.incoming.get())

    async def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self) -> None:
        self.ws = _FakeWebSocket()
        self.connects = 0

    async def ws_connect(self, url: str, **_kwargs: object) -> _FakeWebSocket:
        self.connects += 1
        return self.ws


def _offline_stt(**kwargs: Any) -> stt.STT:
    """An STT whose streams connect to a fake socket instead of the network."""
    instance = stt.STT(api_key="test-key", **kwargs)
    instance._session = _FakeSession()  # type: ignore[assignment]
    return instance


def _audio_input_of(payload: dict[str, Any]) -> dict[str, Any]:
    audio_input = payload["session"]["audio"]["input"]
    assert isinstance(audio_input, dict)
    return audio_input


def _audio_input(instance: stt.STT) -> dict[str, Any]:
    return _audio_input_of(
        stt._session_update(instance._opts).model_dump(by_alias=True, exclude_unset=True)
    )


def _silence() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * 2400, sample_rate=24000, num_channels=1, samples_per_channel=2400
    )


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


async def test_detect_language_falls_back_to_the_last_language(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # detection is the absence of a language, so turning it off has to name one again
    instance = stt.STT(api_key="test-key", model="gpt-live-transcribe", language="en")
    instance.update_options(language="de")

    instance.update_options(detect_language=True)
    assert instance._opts.languages == []
    assert not caplog.records  # asking for detection says everything it needs to

    # turning it off names none, so it falls back to the last one asked for and says so
    instance.update_options(detect_language=False)
    assert instance._opts.languages == ["de"]
    assert "falling back to ['de']" in caplog.text

    # naming one alongside it leaves nothing to guess
    caplog.clear()
    instance.update_options(detect_language=True)
    instance.update_options(detect_language=False, language="es")
    assert instance._opts.languages == ["es"]
    assert not caplog.records

    # and turning it off when it was never on changes nothing
    instance.update_options(detect_language=False)
    assert instance._opts.languages == ["es"]
    assert not caplog.records


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


def _transcription_of(payload: dict[str, Any]) -> dict[str, Any]:
    config = _audio_input_of(payload)["transcription"]
    assert isinstance(config, dict)
    return config


def _sent_transcription(ws: _FakeWebSocket) -> dict[str, Any]:
    return _transcription_of(ws.sent[-1])


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

    assert not stream._reconnect_event.is_set()

    # a language reaches the open connection too, and the transcripts follow it
    instance.update_options(language="fr")
    await _settle(stream)
    assert stream._language == "fr"
    assert _sent_transcription(ws)["languages"] == ["fr"]
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


async def test_switching_to_a_realtime_only_model_needs_the_realtime_transport() -> None:
    # the transcriptions endpoint does not serve these models at all, and the transport cannot
    # change once AgentSession has wrapped a non-streaming STT in a StreamAdapter
    instance = stt.STT(api_key="test-key", model="gpt-transcribe", vad=None)
    assert instance.capabilities.streaming is False

    for model in ("gpt-live-transcribe", "gpt-realtime-whisper"):
        with pytest.raises(ValueError, match="served only over the realtime API"):
            instance.update_options(model=model)
    assert instance._opts.model == "gpt-transcribe"

    # the realtime API serves every model, so that direction stays open
    realtime = stt.STT(api_key="test-key", model="gpt-live-transcribe", vad=None)
    realtime.update_options(model="gpt-transcribe")
    assert realtime._opts.model == "gpt-transcribe"

    # an explicit use_realtime=False keeps that pairing, and it stays usable
    pinned = stt.STT(api_key="test-key", model="gpt-live-transcribe", use_realtime=False, vad=None)
    pinned.update_options(prompt="a support call")
    assert pinned._opts.prompt == "a support call"


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


@pytest.mark.parametrize(
    ("model", "realtime"),
    [
        ("gpt-live-transcribe", True),
        ("gpt-realtime-whisper", True),
        ("gpt-4o-mini-transcribe", False),
        ("whisper-1", False),
    ],
)
def test_use_realtime_defaults_to_the_only_transport_a_model_has(
    model: str, realtime: bool
) -> None:
    instance = stt.STT(api_key="test-key", model=model, vad=None)
    assert instance.capabilities.streaming is realtime

    # an explicit value still wins
    off = stt.STT(api_key="test-key", model=model, vad=None, use_realtime=False)
    assert off.capabilities.streaming is False


def test_a_client_commit_model_loads_the_bundled_vad() -> None:
    # no livekit-plugins-silero dependency: the VAD ships with livekit-agents
    instance = stt.STT(api_key="test-key", model="gpt-live-transcribe")
    assert isinstance(instance._vad, inference.VAD)

    assert stt.STT(api_key="test-key", model="gpt-4o-mini-transcribe")._vad is None


async def test_live_transcribe_omits_turn_detection() -> None:
    # the model rejects any turn_detection config and needs a client-side commit
    instance = stt.STT(api_key="test-key", model="gpt-live-transcribe", use_realtime=True, vad=None)

    assert "turn_detection" not in _audio_input(instance)

    # every other model still gets server-side VAD
    other = stt.STT(api_key="test-key", model="gpt-4o-mini-transcribe", use_realtime=True)
    assert _audio_input(other)["turn_detection"]["type"] == "server_vad"


async def test_turn_detection_without_a_type_is_still_sent() -> None:
    # SessionTurnDetection leaves every key optional, so a caller can tune one setting alone
    instance = stt.STT(
        api_key="test-key",
        model="gpt-4o-mini-transcribe",
        use_realtime=True,
        turn_detection={"silence_duration_ms": 800},
    )

    assert _audio_input(instance)["turn_detection"] == {
        "type": "server_vad",
        "silence_duration_ms": 800,
    }

    instance.update_options(turn_detection={"type": "semantic_vad", "eagerness": "low"})
    assert _audio_input(instance)["turn_detection"] == {
        "type": "semantic_vad",
        "eagerness": "low",
    }


class _OneShotVAD(vad.VAD):
    """A VAD that closes one segment per stream, on the first frame it sees."""

    def __init__(self) -> None:
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.1))

    def stream(self) -> vad.VADStream:
        return _OneShotVADStream(self)


class _OneShotVADStream(vad.VADStream):
    async def _main_task(self) -> None:
        done = False
        # the input has to be drained until it ends, or the stream closes under its consumer
        async for frame in self._input_ch:
            if done or not isinstance(frame, rtc.AudioFrame):
                continue
            done = True
            for type in (vad.VADEventType.START_OF_SPEECH, vad.VADEventType.END_OF_SPEECH):
                self._event_ch.send_nowait(
                    vad.VADEvent(
                        type=type,
                        samples_index=0,
                        timestamp=0.0,
                        speech_duration=0.0,
                        silence_duration=0.0,
                    )
                )


def _of_type(ws: _FakeWebSocket, type: str) -> list[dict[str, Any]]:
    return [payload for payload in ws.sent if payload.get("type") == type]


async def _wait_for(what: str, ready: Callable[[], bool]) -> None:
    for _ in range(200):
        if ready():
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"timed out waiting for {what}")


async def test_the_vad_stops_committing_once_the_server_endpoints() -> None:
    # the client VAD and server-side turn detection would otherwise both close the segment
    instance = _offline_stt(model="gpt-live-transcribe", vad=_OneShotVAD())
    stream = instance.stream()
    ws = await _connected(stream)
    ends = 0

    async def collect() -> None:
        nonlocal ends
        async for ev in stream:
            if ev.type == SpeechEventType.END_OF_SPEECH:
                ends += 1

    collector = asyncio.create_task(collect())

    stream.push_frame(_silence())
    await _wait_for("the first commit", lambda: len(_of_type(ws, "input_audio_buffer.commit")) == 1)

    instance.update_options(model="gpt-4o-mini-transcribe")
    await _wait_for("the rebuilt connection", lambda: len(_of_type(ws, "session.update")) == 2)
    assert "turn_detection" in _audio_input_of(_of_type(ws, "session.update")[-1])

    # the rebuilt connection has its own VAD stream, so the segment closes a second time
    stream.push_frame(_silence())
    await _wait_for("the second segment", lambda: ends == 2)
    await asyncio.sleep(0.05)  # a commit would follow the segment straight away
    assert len(_of_type(ws, "input_audio_buffer.commit")) == 1

    await stream.aclose()
    await collector


async def test_an_update_during_connection_setup_is_not_lost() -> None:
    instance = _offline_stt(model="gpt-live-transcribe")
    session = _FakeSession()
    instance._session = session  # type: ignore[assignment]
    release = asyncio.Event()
    session.ws.block_first = release  # hold the stream inside its setup send

    stream = instance.stream()
    ws = session.ws
    for _ in range(200):  # wait until the stream is suspended inside that send
        if ws.block_first is None:
            break
        await asyncio.sleep(0.005)
    else:
        raise AssertionError("the setup send never started")
    assert ws.sent == []

    instance.update_options(keywords=["Acme Corp"])
    release.set()
    await _settle(stream)

    # the change is applied after the setup config rather than dropped or overtaken by it
    assert [_transcription_of(p).get("keywords") for p in ws.sent] == [[], ["Acme Corp"]]

    await stream.aclose()


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


@pytest.mark.parametrize(
    ("change", "connects"),
    [
        # gateways route on the ?model= in the upgrade URL, so a reused socket keeps the old one
        ({"model": "gpt-4o-mini-transcribe"}, 2),
        ({"detect_language": True}, 2),  # `languages` cannot be cleared on an open session
        ({"keywords": ["Acme Corp"]}, 1),  # a session.update on acquire carries this one
    ],
)
async def test_a_rebuild_drops_the_idle_pooled_connection(
    change: dict[str, Any], connects: int
) -> None:
    instance = _offline_stt(model="gpt-live-transcribe")
    session = _FakeSession()
    instance._session = session  # type: ignore[assignment]
    # a socket the pool holds with no stream attached, as between two speech sessions
    instance._pool.put(await instance._pool.get(timeout=5))
    assert not instance._streams

    instance.update_options(**change)

    stream = instance.stream()
    await _connected(stream)
    assert session.connects == connects

    await stream.aclose()


async def test_a_stream_with_its_own_language_takes_its_own_connection() -> None:
    # a pooled socket carries whatever language the stream before it set
    instance = _offline_stt(model="gpt-live-transcribe", language="en")
    session = _FakeSession()
    instance._session = session  # type: ignore[assignment]
    instance._pool.put(await instance._pool.get(timeout=5))

    stream = instance.stream(language=[])

    assert stream._opts.languages == []
    assert instance._opts.languages == ["en"]  # the STT default is untouched
    await _connected(stream)
    assert session.connects == 2

    await stream.aclose()


async def test_a_stream_language_stays_out_of_the_streams_already_open() -> None:
    instance = _offline_stt(model="gpt-live-transcribe", language="en")
    open_stream = instance.stream()
    ws = await _connected(open_stream)

    later = instance.stream(language="fr")
    await _connected(later)

    assert open_stream._opts.languages == ["en"]
    assert open_stream._language == "en"
    assert later._opts.languages == ["fr"]
    # the new stream sets its language with a session.update, so the connection the open one
    # is reading from stays up
    assert not ws.closed

    await open_stream.aclose()
    await later.aclose()


async def test_a_stream_switches_its_own_language() -> None:
    instance = _offline_stt(model="gpt-live-transcribe", language="en")
    stream = instance.stream()
    ws = await _connected(stream)
    other = instance.stream()

    stream.update_options(language="fr")
    await _settle(stream)

    # the stream carries its own language over its own connection, and nothing else moves
    assert _sent_transcription(ws)["languages"] == ["fr"]
    assert stream._language == "fr"
    assert other._opts.languages == ["en"]
    assert instance._opts.languages == ["en"]

    # and it is pinned, so an STT change that names no language leaves it alone
    instance.update_options(keywords=["Acme Corp"])
    assert stream._opts.languages == ["fr"]

    await stream.aclose()
    await other.aclose()


async def test_an_stt_language_change_overrides_a_stream_of_its_own() -> None:
    instance = _offline_stt(model="gpt-live-transcribe", language="en")
    pinned = instance.stream(language="fr")
    plain = instance.stream()

    # a change that names no language leaves the pinned stream on its own
    instance.update_options(keywords=["Acme Corp"])
    assert pinned._opts.languages == ["fr"]
    assert pinned._opts.keywords == ["Acme Corp"]  # the rest still reaches it
    assert plain._opts.languages == ["en"]

    # naming one takes the stream back
    instance.update_options(language="de")
    assert pinned._opts.languages == ["de"]
    assert plain._opts.languages == ["de"]

    # so does asking for detection
    instance.update_options(language=[])
    assert pinned._opts.languages == []

    await pinned.aclose()
    await plain.aclose()


async def test_session_keyterms_reach_the_next_connection() -> None:
    instance = stt.STT(api_key="test-key", model="gpt-live-transcribe")
    instance._update_session_keyterms(["Acme Corp"])

    config = await _transcription_config(instance)

    assert config["keywords"] == ["Acme Corp"]


async def _transcription_form(instance: stt.STT, captured: list[httpx.Request]) -> str:
    event = await instance._recognize_impl(_silence(), conn_options=DEFAULT_API_CONNECT_OPTIONS)
    assert event.alternatives[0].text == "hello"
    return captured[0].read().decode("utf-8", errors="replace")


def _mock_client(
    captured: list[httpx.Request], body: dict[str, Any] | None = None
) -> openai.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=body or {"text": "hello"})

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


@pytest.mark.parametrize(
    ("detected", "dominant"),
    [
        ([{"code": "fr"}], "fr"),  # what the model heard wins over the hint
        ([], "en"),  # nothing detected reliably, so the hint stands
        ([{"code": "en"}, {"code": "fr"}], "en"),  # code-switched: the dominant code comes first
    ],
)
async def test_file_transcription_reports_the_detected_language(
    detected: list[dict[str, str]], dominant: str
) -> None:
    captured: list[httpx.Request] = []
    instance = stt.STT(
        model="gpt-transcribe",
        client=_mock_client(captured, {"text": "hello", "languages": detected}),
        language="en",
    )

    event = await instance._recognize_impl(_silence(), conn_options=DEFAULT_API_CONNECT_OPTIONS)

    assert event.alternatives[0].language == dominant


async def test_realtime_reports_the_detected_language() -> None:
    # the hint is a list, so there is no single code to fall back on
    instance = _offline_stt(model="gpt-live-transcribe", language=["en", "fr"], vad=None)
    stream = instance.stream()
    ws = await _connected(stream)
    assert stream._language == ""

    ws.incoming.put_nowait(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_1",
            "transcript": "bonjour, hello",
            "languages": [{"code": "fr"}, {"code": "en"}],
        }
    )

    final: SpeechEvent | None = None
    try:
        async for event in stream:
            if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                final = event
                break
    finally:
        await stream.aclose()

    assert final is not None
    assert final.alternatives[0].text == "bonjour, hello"
    assert final.alternatives[0].language == "fr"


async def test_file_transcription_earlier_model_keeps_singular_language() -> None:
    captured: list[httpx.Request] = []
    instance = stt.STT(model="whisper-1", client=_mock_client(captured), language="en-US")

    body = await _transcription_form(instance, captured)

    assert 'name="language"\r\n\r\nen' in body
    assert 'name="languages[]"' not in body
