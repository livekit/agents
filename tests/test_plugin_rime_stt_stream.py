import asyncio
from types import SimpleNamespace

import pytest
import rimelabs_sdk as sdk

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, stt, vad
from livekit.plugins import rime

pytestmark = pytest.mark.unit


class DetectorStream:
    def __init__(self):
        self.events = asyncio.Queue()
        self.closed = False

    def push_frame(self, frame):
        pass

    def flush(self):
        pass

    def end_input(self):
        self.events.put_nowait(None)

    async def aclose(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        value = await self.events.get()
        if value is None:
            raise StopAsyncIteration
        return value

    def send(self, kind, data):
        frame = rtc.AudioFrame(
            data=data, sample_rate=16000, num_channels=1, samples_per_channel=len(data) // 2
        )
        self.events.put_nowait(
            SimpleNamespace(
                type=kind,
                frames=[frame],
                silence_duration=0.5,
                inference_duration=0,
            )
        )


class SDKStream:
    request_id = "test-request"

    def __init__(self, source, options, fail):
        self.source = source
        self.options = options
        self.audio = []
        self.closed = False
        self.fail = fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def updates(self):
        text = ""
        async for chunk in self.source:
            self.audio.append(chunk)
            delta = "Ice" if not text else "\ncream"
            text += delta
            if self.options["output_contract"] == sdk.TranscriptionOutputContract.APPEND_FRAGMENTS:
                yield sdk.TranscriptionDelta(text=delta)
            else:
                yield sdk.TranscriptionPartial(text=text)
            if self.fail:
                raise sdk.RimeUnavailableError(
                    "private-provider-content", request_id="failed-request"
                )
        yield sdk.TranscriptionFinal(text=text, language="en")

    def __aiter__(self):
        return self.updates()


@pytest.fixture
def setup(monkeypatch):
    streams = []
    detector = DetectorStream()
    closed = []
    events = []
    fail = [False]

    def stream(source, **options):
        result = SDKStream(source, options, fail[0])
        streams.append(result)
        return result

    async def close():
        closed.append(True)

    monkeypatch.setattr(
        sdk,
        "Rime",
        lambda **kw: SimpleNamespace(
            stt=SimpleNamespace(stream=stream, transcribe=lambda: None), close=close
        ),
    )
    owner = rime.STT(
        vad=SimpleNamespace(stream=lambda: detector),
        output_contract="append_fragments",
    )
    return owner, detector, streams, events, fail, closed


async def receive(stream, kind):
    while True:
        event = await asyncio.wait_for(anext(stream), 2)
        if event.type == kind:
            return event


@pytest.mark.parametrize("backend", ["sdk_append", "sdk_revised"])
async def test_live_audio_prefix_deltas_and_final(setup, backend):
    owner, detector, streams, events, _, _ = setup
    owner._output_contract = "append_fragments" if backend == "sdk_append" else "revised_hypotheses"
    stream = owner.stream(conn_options=APIConnectOptions(timeout=120))
    try:
        # INFERENCE before START is already included in START's prefix.
        detector.send(vad.VADEventType.INFERENCE_DONE, b"xx")
        detector.send(vad.VADEventType.START_OF_SPEECH, b"aabb")
        first = await receive(stream, stt.SpeechEventType.INTERIM_TRANSCRIPT)
        assert first.alternatives[0].text == "Ice"
        assert not streams[0].closed  # text arrived while input remains open
        detector.send(vad.VADEventType.INFERENCE_DONE, b"cc")
        second = await receive(stream, stt.SpeechEventType.INTERIM_TRANSCRIPT)
        assert second.alternatives[0].text == "Ice\ncream"
        detector.send(vad.VADEventType.END_OF_SPEECH, b"aabbcc")
        final = await receive(stream, stt.SpeechEventType.FINAL_TRANSCRIPT)
        assert final.alternatives[0].text == "Ice\ncream"
        assert streams[0].audio == [b"aabb", b"cc"]  # no duplicate END buffer
        assert streams[0].closed
        assert final.request_id == "test-request"

        stream.end_input()
        assert all(
            e.type == stt.SpeechEventType.RECOGNITION_USAGE for e in [e async for e in stream]
        )
    finally:
        await owner.aclose()
    assert detector.closed


async def test_options_snapshot_and_second_utterance(setup):
    owner, detector, streams, _, _, _ = setup
    owner.update_options(context_terms=["Rime"])
    owner._update_session_keyterms(["Rime", "LiveKit"])
    stream = owner.stream()
    try:
        detector.send(vad.VADEventType.START_OF_SPEECH, b"aa")
        await receive(stream, stt.SpeechEventType.INTERIM_TRANSCRIPT)
        owner.update_options(mode="verbatim", language="es", context_terms=["New"])
        detector.send(vad.VADEventType.END_OF_SPEECH, b"aa")
        await receive(stream, stt.SpeechEventType.FINAL_TRANSCRIPT)
        detector.send(vad.VADEventType.START_OF_SPEECH, b"bb")
        await receive(stream, stt.SpeechEventType.INTERIM_TRANSCRIPT)
        detector.send(vad.VADEventType.END_OF_SPEECH, b"bb")
        await receive(stream, stt.SpeechEventType.FINAL_TRANSCRIPT)
        assert streams[0].options["mode"] == sdk.TranscriptionMode.WRITTEN
        assert streams[0].options["context_terms"] == ("Rime", "LiveKit")
        assert streams[1].options["mode"] == sdk.TranscriptionMode.VERBATIM
        assert streams[1].options["language"] == "es"
        assert streams[1].options["context_terms"] == ("New", "Rime", "LiveKit")
    finally:
        await owner.aclose()


async def test_failure_before_vad_end_is_not_retried(setup):
    owner, detector, streams, events, fail, _ = setup
    fail[0] = True
    stream = owner.stream(conn_options=APIConnectOptions(max_retry=3))
    try:
        detector.send(vad.VADEventType.START_OF_SPEECH, b"aa")
        observed = []
        with pytest.raises(APIStatusError) as caught:
            async for event in stream:
                observed.append(event)
        assert caught.value.request_id == "failed-request"
        assert not caught.value.retryable
        assert "private-provider-content" not in str(caught.value)
        assert not any(e.type == stt.SpeechEventType.FINAL_TRANSCRIPT for e in observed)
        assert len(streams) == 1
        assert not any(e.get("kind") == "final" for e in events)
        assert streams[0].closed
    finally:
        await owner.aclose()


async def test_stop_cancels_active_rpc(setup):
    owner, detector, streams, events, _, closed = setup
    stream = owner.stream()
    detector.send(vad.VADEventType.START_OF_SPEECH, b"aa")
    await receive(stream, stt.SpeechEventType.INTERIM_TRANSCRIPT)
    await owner.aclose()
    assert streams[0].closed and detector.closed and closed
    assert not owner._streams
    assert not any(e.get("kind") == "final" for e in events)


async def test_silence_does_not_start_rpc(setup):
    owner, detector, streams, _, _, _ = setup
    stream = owner.stream()
    detector.send(vad.VADEventType.INFERENCE_DONE, b"aa")
    stream.end_input()
    assert [e async for e in stream] == []
    assert not streams
    await owner.aclose()


async def test_end_input_finishes_active_utterance_without_vad_end(setup):
    owner, detector, streams, _, _, _ = setup
    stream = owner.stream()
    try:
        detector.send(vad.VADEventType.START_OF_SPEECH, b"aa")
        await receive(stream, stt.SpeechEventType.INTERIM_TRANSCRIPT)
        stream.end_input()
        events = [event async for event in stream]
        assert [e.type for e in events] == [
            stt.SpeechEventType.END_OF_SPEECH,
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            stt.SpeechEventType.RECOGNITION_USAGE,
        ]
        assert events[1].alternatives[0].text == "Ice"
        assert streams[0].audio == [b"aa"]
    finally:
        await owner.aclose()


async def test_backlog_is_bounded_in_bytes_and_does_not_finalize(setup):
    owner, detector, streams, _, _, _ = setup
    stream = owner.stream()
    detector.send(vad.VADEventType.START_OF_SPEECH, bytes(320002))
    try:
        from livekit.agents import APIError

        with pytest.raises(APIError, match="backlog"):
            await receive(stream, stt.SpeechEventType.FINAL_TRANSCRIPT)
        assert not streams
        assert detector.closed
    finally:
        await owner.aclose()


async def test_language_override_and_silence_final(setup, monkeypatch):
    owner, detector, streams, _, _, _ = setup

    async def silence(self):
        async for chunk in self.source:
            self.audio.append(chunk)
        yield sdk.TranscriptionFinal(text="", language="es")

    monkeypatch.setattr(SDKStream, "updates", silence)
    stream = owner.stream(language="es")
    try:
        detector.send(vad.VADEventType.START_OF_SPEECH, b"aa")
        detector.send(vad.VADEventType.END_OF_SPEECH, b"aa")
        final = await receive(stream, stt.SpeechEventType.FINAL_TRANSCRIPT)
        assert final.alternatives[0].text == ""
        assert final.alternatives[0].language == "es"
        assert streams[0].options["language"] == "es"
    finally:
        await owner.aclose()


async def test_revised_hypotheses_replace_previous_text(setup, monkeypatch):
    owner, detector, streams, _, _, _ = setup
    owner._output_contract = "revised_hypotheses"

    async def revised(self):
        async for chunk in self.source:
            self.audio.append(chunk)
        yield sdk.TranscriptionPartial(text="I scream")
        yield sdk.TranscriptionPartial(text="Ice cream")
        yield sdk.TranscriptionFinal(text="Ice cream.", language="en")

    monkeypatch.setattr(SDKStream, "updates", revised)
    stream = owner.stream()
    try:
        detector.send(vad.VADEventType.START_OF_SPEECH, b"aa")
        detector.send(vad.VADEventType.END_OF_SPEECH, b"aa")
        texts = []
        while True:
            event = await asyncio.wait_for(anext(stream), 2)
            if event.alternatives:
                texts.append(event.alternatives[0].text)
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                break
        assert texts == ["I scream", "Ice cream", "Ice cream."]
    finally:
        await owner.aclose()


async def test_flush_restarts_vad_and_resamples_live_frames(setup, monkeypatch):
    owner, _, streams, _, _, _ = setup
    detectors = []

    class LiveDetector(DetectorStream):
        def __init__(self):
            super().__init__()
            self.started = False
            detectors.append(self)

        def push_frame(self, frame):
            assert frame.sample_rate == 16000
            kind = (
                vad.VADEventType.INFERENCE_DONE
                if self.started
                else vad.VADEventType.START_OF_SPEECH
            )
            self.started = True
            self.send(kind, bytes(frame.data))

    owner._vad = SimpleNamespace(stream=LiveDetector)
    stream = owner.stream()
    frame = rtc.AudioFrame(
        data=bytes(9600), sample_rate=48000, num_channels=1, samples_per_channel=4800
    )
    try:
        stream.push_frame(frame)
        stream.flush()
        stream.push_frame(frame)
        stream.end_input()
        events = [event async for event in stream]
        assert len([e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]) == 2
        assert len(streams) == 2
        assert all(sum(map(len, s.audio)) == 3200 for s in streams)
        assert all(d.closed for d in detectors)
    finally:
        await owner.aclose()


async def wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), 2)


@pytest.mark.parametrize("contract", ["append_fragments", "revised_hypotheses"])
async def test_later_utterance_uploads_while_first_final_waits(setup, monkeypatch, contract):
    owner, detector, streams, _, _, _ = setup
    owner._output_contract = contract
    release = asyncio.Event()
    first_eof = asyncio.Event()
    second_done = asyncio.Event()

    async def delayed(self):
        first = self is streams[0]
        self.request_id = "first" if first else "second"
        text = ""
        async for chunk in self.source:
            self.audio.append(chunk)
            text += "a" if first else "b"
            if contract == "append_fragments":
                yield sdk.TranscriptionDelta(text="a" if first else "b")
            else:
                yield sdk.TranscriptionPartial(text=text)
        if first:
            first_eof.set()
            await release.wait()
        yield sdk.TranscriptionFinal(text=text, language=self.options["language"])
        if not first:
            second_done.set()

    monkeypatch.setattr(SDKStream, "updates", delayed)
    stream = owner.stream(conn_options=APIConnectOptions(timeout=120))
    observed = []

    async def collect():
        async for event in stream:
            observed.append(event)

    reader = asyncio.create_task(collect())
    try:
        detector.send(vad.VADEventType.START_OF_SPEECH, b"aa")
        await wait_until(lambda: streams and streams[0].audio)
        detector.send(vad.VADEventType.END_OF_SPEECH, b"aa")
        await asyncio.wait_for(first_eof.wait(), 2)
        owner.update_options(mode="verbatim", language="es", context_terms=["Second"])
        detector.send(vad.VADEventType.START_OF_SPEECH, bytes(32000))
        await wait_until(lambda: len(streams) == 2 and len(streams[1].audio) == 1)
        # Eleven seconds consumed while the previous final is still unavailable.
        for count in range(2, 12):
            detector.send(vad.VADEventType.INFERENCE_DONE, bytes(32000))
            await wait_until(lambda count=count: len(streams[1].audio) == count)
        detector.send(vad.VADEventType.END_OF_SPEECH, bytes(352000))
        await asyncio.wait_for(second_done.wait(), 2)
        await wait_until(
            lambda: sum(e.type == stt.SpeechEventType.END_OF_SPEECH for e in observed) == 2
        )
        assert sum(e.type == stt.SpeechEventType.START_OF_SPEECH for e in observed) == 2
        assert not any(e.type == stt.SpeechEventType.FINAL_TRANSCRIPT for e in observed)
        assert streams[1].options["language"] == "es"
        assert streams[1].options["mode"] == sdk.TranscriptionMode.VERBATIM
        assert streams[1].options["context_terms"] == ("Second",)
        ends = [e.speech_end_time for e in observed if e.type == stt.SpeechEventType.END_OF_SPEECH]
        stream.end_input()
        release.set()
        await asyncio.wait_for(reader, 2)
        finals = [e for e in observed if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
        assert [e.request_id for e in finals] == ["first", "second"]
        assert [e.alternatives[0].text for e in finals] == ["a", "b" * 11]
        assert [e.speech_end_time for e in finals] == ends
        usage = [e.recognition_usage.audio_duration for e in observed if e.recognition_usage]
        assert usage == pytest.approx([1 / 16000, 11])
        assert all(s.closed for s in streams)
    finally:
        await owner.aclose()
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)


@pytest.mark.parametrize("failing", [0, 1, None])
async def test_parallel_requests_cancel_on_error_or_close(setup, monkeypatch, failing):
    owner, detector, streams, _, _, _ = setup
    fail_now = asyncio.Event()

    async def delayed(self):
        index = streams.index(self)
        async for chunk in self.source:
            self.audio.append(chunk)
        await fail_now.wait()
        if index == failing:
            raise sdk.RimeUnavailableError("private-provider-content", request_id="failed-request")
        await asyncio.Event().wait()
        yield  # keep this an async generator

    monkeypatch.setattr(SDKStream, "updates", delayed)
    stream = owner.stream()
    observed = []

    async def collect():
        async for event in stream:
            observed.append(event)

    reader = asyncio.create_task(collect())
    try:
        for count in (1, 2):
            detector.send(vad.VADEventType.START_OF_SPEECH, b"aa")
            detector.send(vad.VADEventType.END_OF_SPEECH, b"aa")
            await wait_until(lambda count=count: len(streams) == count and streams[-1].audio)
        if failing is None:
            await owner.aclose()
        else:
            fail_now.set()
            with pytest.raises(APIStatusError):
                await asyncio.wait_for(reader, 2)
        assert all(s.closed for s in streams)
        assert detector.closed
        assert not any(e.type == stt.SpeechEventType.FINAL_TRANSCRIPT for e in observed)
    finally:
        await owner.aclose()
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)


async def test_flush_accepts_new_audio_before_previous_final(setup, monkeypatch):
    owner, _, streams, _, _, _ = setup
    release = asyncio.Event()
    first_eof = asyncio.Event()
    second_eof = asyncio.Event()

    class LiveDetector(DetectorStream):
        def push_frame(self, frame):
            self.send(vad.VADEventType.START_OF_SPEECH, bytes(frame.data))

    owner._vad = SimpleNamespace(stream=LiveDetector)

    async def delayed(self):
        async for chunk in self.source:
            self.audio.append(chunk)
        if self is streams[0]:
            first_eof.set()
            await release.wait()
        else:
            second_eof.set()
        yield sdk.TranscriptionFinal(text=str(streams.index(self)), language="en")

    monkeypatch.setattr(SDKStream, "updates", delayed)
    stream = owner.stream()
    frame = rtc.AudioFrame(
        data=bytes(3200), sample_rate=16000, num_channels=1, samples_per_channel=1600
    )
    observed = []

    async def collect():
        async for event in stream:
            observed.append(event)

    reader = asyncio.create_task(collect())
    try:
        stream.push_frame(frame)
        stream.flush()
        await asyncio.wait_for(first_eof.wait(), 2)
        stream.push_frame(frame)
        stream.end_input()
        await asyncio.wait_for(second_eof.wait(), 2)
        assert not any(e.type == stt.SpeechEventType.FINAL_TRANSCRIPT for e in observed)
        release.set()
        await asyncio.wait_for(reader, 2)
        assert [
            e.alternatives[0].text
            for e in observed
            if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ] == ["0", "1"]
    finally:
        await owner.aclose()
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)


async def test_pending_utterances_are_bounded(setup, monkeypatch):
    from livekit.agents import APIError

    owner, detector, streams, _, _, _ = setup

    async def delayed(self):
        async for chunk in self.source:
            self.audio.append(chunk)
        await asyncio.Event().wait()
        yield

    monkeypatch.setattr(SDKStream, "updates", delayed)
    stream = owner.stream()

    async def collect():
        return [event async for event in stream]

    reader = asyncio.create_task(collect())
    try:
        for count in range(1, 9):
            detector.send(vad.VADEventType.START_OF_SPEECH, b"aa")
            detector.send(vad.VADEventType.END_OF_SPEECH, b"aa")
            await wait_until(lambda count=count: len(streams) == count and streams[-1].audio)
        detector.send(vad.VADEventType.START_OF_SPEECH, b"aa")
        with pytest.raises(APIError, match="pending utterance limit"):
            await asyncio.wait_for(reader, 2)
        assert len(streams) == 8
        assert all(s.closed for s in streams)
        assert detector.closed
    finally:
        await owner.aclose()
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
