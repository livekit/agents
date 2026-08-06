"""`inference.STT` must not claim word-aligned transcripts for models with no word data.

The gateway's Cartesia Ink-2 translator emits ``Words: []`` with ``Start`` = the
audio-time offset and ``Duration`` = billed audio bytes / bytes-per-second — a billing
figure, not the utterance span. Cartesia's Turns API sends no word timings to forward.

That flag is load-bearing: `AgentActivity._resolve_interruption_detection` uses it as the
`can_gatekeep` test for enabling adaptive interruption, which then disables the fast
VAD interruption path once the backchannel boundary expires. So a false claim costs
barge-in latency on exactly the models least able to make up for it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from livekit.agents import (
    Agent,
    AgentSession,
    EndpointingOptions,
    InterruptionOptions,
    TurnHandlingOptions,
    inference,
)
from livekit.agents.inference.stt import SpeechStream as InferenceSpeechStream
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import AudioRecognition

from .fake_llm import FakeLLM
from .fake_stt import FakeUserSpeech
from .fake_tts import FakeTTS
from .fake_turn_stt import ScriptedTurn, TurnScriptedSTT
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

# What the gateway actually puts on the wire for a cartesia/ink-2 final transcript.
# See agent-gateway/pkg/provider/stt/cartesiaturns/translator.go
# emitTranscriptWithBytes: Start = GetAudioTimeOffset().Seconds(),
# Duration = audioBytes/bytesPerSec (final only), Words = []interfaces.Word{}.
GATEWAY_INK2_FINAL = {
    "transcript": "are you open on sunday",
    "confidence": 1.0,
    "start": 0.0,
    "duration": 12.5,  # audio billed since the previous turn.end, not this utterance
    "words": [],
    "language": "en",
}


@pytest.fixture
def _fake_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "fake")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "fakesecret")
    monkeypatch.setenv("CARTESIA_API_KEY", "fake")


def test_inference_and_plugin_agree_about_ink2(_fake_credentials: None) -> None:
    """The same Cartesia model must report the same alignment through either integration.

    Verified against the live Turns API: every event is {type, transcript, turn_id,
    request_id}, on both the version the plugin pins and the one the gateway pins, and
    neither `add_timestamps` nor `timestamp_granularities[]` changes that.
    """
    cartesia = pytest.importorskip("livekit.plugins.cartesia")

    gateway_stt = inference.STT(model="cartesia/ink-2")
    plugin_stt = cartesia.STT(model="ink-2")

    assert plugin_stt.capabilities.aligned_transcript is False
    assert gateway_stt.capabilities.aligned_transcript is False


def test_models_that_do_send_words_keep_their_claim(_fake_credentials: None) -> None:
    """The fix must not strip alignment from the providers that genuinely forward words."""
    assert inference.STT(model="cartesia/ink-whisper").capabilities.aligned_transcript == "word"
    assert inference.STT(model="deepgram/nova-3").capabilities.aligned_transcript == "word"
    assert inference.STT(
        model="assemblyai/universal-streaming"
    ).capabilities.aligned_transcript == ("word")
    # the provider is resolved server-side per language, so alignment can't be promised
    assert inference.STT(model="auto").capabilities.aligned_transcript is False
    assert inference.STT(model="inworld/inworld-stt-1").capabilities.aligned_transcript is False


def test_alignment_is_recomputed_when_the_model_changes(_fake_credentials: None) -> None:
    """`update_options` recomputes it, as it already does for keyterms and chat_context."""
    stt_impl = inference.STT(model="deepgram/nova-3")
    assert stt_impl.capabilities.aligned_transcript == "word"

    stt_impl.update_options(model="cartesia/ink-2")
    assert stt_impl.capabilities.aligned_transcript is False


def test_unknown_models_do_not_claim_alignment(_fake_credentials: None) -> None:
    stt_impl = inference.STT(model="new-provider/new-turn-model")
    assert stt_impl.capabilities.aligned_transcript is False


@pytest.mark.parametrize(
    ("fallback", "expected"),
    [
        pytest.param("cartesia/ink-whisper", "word", id="aligned-fallback"),
        pytest.param("cartesia/ink-2", False, id="unaligned-fallback"),
        pytest.param("new-provider/new-turn-model", False, id="unknown-fallback"),
    ],
)
def test_fallback_models_constrain_alignment_claim(
    fallback: str, expected: object, _fake_credentials: None
) -> None:
    stt_impl = inference.STT(model="deepgram/nova-3", fallback=fallback)
    assert stt_impl.capabilities.aligned_transcript == expected


def test_alignment_update_still_accounts_for_fallback(_fake_credentials: None) -> None:
    stt_impl = inference.STT(model="cartesia/ink-2", fallback="new-provider/new-turn-model")

    stt_impl.update_options(model="deepgram/nova-3")

    assert stt_impl.capabilities.aligned_transcript is False


def test_gateway_ink2_payload_carries_no_word_alignment() -> None:
    """The advertised word alignment is not in the data the gateway sends."""
    stream = InferenceSpeechStream.__new__(InferenceSpeechStream)
    stream._start_time_offset = 0.0
    stream._opts = SimpleNamespace(language="en")

    data = stream._build_speech_data(GATEWAY_INK2_FINAL)

    # advertised as "word"-aligned, but there are no words to align to
    assert data.words == []
    # and the span is the billing figure, ~12.5s for a ~2s utterance
    assert data.start_time == 0.0
    assert data.end_time == 12.5


def _recognition_awaiting_flush(
    *, ignore_until: float, input_started_at: float
) -> AudioRecognition:
    """An AudioRecognition mid-ignore-window, as it sits just after agent speech ends."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._interruption_enabled = True
    ar._agent_speaking = False
    ar._ignore_user_transcript_until = ignore_until
    # _input_started_at is a read-only property over the STT pipeline
    ar._stt_pipeline = SimpleNamespace(input_started_at=input_started_at)
    ar._agent_speech_started_at = input_started_at
    return ar


def _final(*, start_time: float, end_time: float) -> SpeechEvent:
    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            SpeechData(
                text="are you open on sunday",
                language="en",
                start_time=start_time,
                end_time=end_time,
            )
        ],
    )


def test_gateway_timestamps_defeat_the_transcript_gate() -> None:
    """Adaptive interruption's transcript gate cannot act on the gateway's Ink-2 output.

    `_should_hold_stt_event` requires ``start_time > 0``, and the gateway pins ``start`` to
    the audio-time offset — 0 until a reconnect. So the half of adaptive interruption that
    is supposed to gatekeep transcripts is inert for this model, even though its
    ``aligned_transcript`` claim is what switched adaptive on.
    """
    import time

    now = time.time()
    ar = _recognition_awaiting_flush(ignore_until=now + 1.0, input_started_at=now - 5.0)

    gateway_shaped = _final(start_time=0.0, end_time=12.5)
    properly_aligned = _final(start_time=4.5, end_time=5.2)

    assert ar._should_hold_stt_event(gateway_shaped) is False
    assert ar._should_hold_stt_event(properly_aligned) is True


def test_reconnect_offset_makes_the_gate_use_a_billing_span() -> None:
    """After a reconnect the gate does engage — on a window derived from billed bytes.

    `RecognizeStream` accumulates `start_time_offset` across retries, so post-reconnect
    ``start_time > 0`` and the hold decision is made with ``end_time - start_time`` equal
    to the audio billed since the previous turn.
    """
    import time

    now = time.time()
    ar = _recognition_awaiting_flush(ignore_until=now + 1.0, input_started_at=now - 5.0)

    # offset 4.5s after a reconnect; duration is still the 12.5s billing figure
    post_reconnect = _final(start_time=4.5, end_time=17.0)
    assert ar._should_hold_stt_event(post_reconnect) is True
    assert (
        post_reconnect.alternatives[0].end_time - post_reconnect.alternatives[0].start_time == 12.5
    )


def _build_session(*, aligned_transcript: object, mode: str | None = None) -> AgentSession:
    speech = FakeUserSpeech(start_time=0.5, end_time=2.0, transcript="hello", stt_delay=0.0)
    interruption = InterruptionOptions(resume_false_interruption=False)
    if mode is not None:
        interruption["mode"] = mode  # type: ignore[typeddict-item]
    return AgentSession[None](
        vad=FakeVAD(fake_user_speeches=[speech]),
        stt=TurnScriptedSTT(
            turns=[ScriptedTurn(speech_start=0.55, final_at=2.6, final_text="hello")],
            aligned_transcript=aligned_transcript,  # type: ignore[arg-type]
        ),
        llm=FakeLLM(fake_responses=[]),
        tts=FakeTTS(fake_responses=[]),
        turn_handling=TurnHandlingOptions(
            turn_detection="stt",
            endpointing=EndpointingOptions(min_delay=0.2),
            # mode absent means the session auto-detects
            interruption=interruption,
        ),
        aec_warmup_duration=None,
    )


@pytest.mark.parametrize(
    ("aligned_transcript", "expect_adaptive"),
    [
        pytest.param("word", True, id="claims-word-alignment"),
        pytest.param(False, False, id="reports-no-alignment"),
    ],
)
def test_alignment_claim_alone_enables_adaptive_interruption(
    aligned_transcript: object,
    expect_adaptive: bool,
    monkeypatch: pytest.MonkeyPatch,
    _fake_credentials: None,
) -> None:
    """Only the capability flag differs, yet it decides whether adaptive interruption runs.

    Dev mode (`lk agent dev` / `console`) and LiveKit Cloud hosted agents auto-enable
    adaptive; plain production leaves it off unless `interruption.mode` is set.
    """
    monkeypatch.setenv("LIVEKIT_DEV_MODE", "1")

    session = _build_session(aligned_transcript=aligned_transcript)
    # AgentActivity resolves interruption detection in __init__, before any I/O starts
    activity = AgentActivity(Agent(instructions="You are a helpful assistant."), session)

    assert activity._interruption_detection_enabled is expect_adaptive


def test_adaptive_stays_off_in_plain_production(
    monkeypatch: pytest.MonkeyPatch, _fake_credentials: None
) -> None:
    """The blast radius is dev mode and hosted agents, not every deployment."""
    monkeypatch.delenv("LIVEKIT_DEV_MODE", raising=False)
    monkeypatch.delenv("LIVEKIT_REMOTE_EOT_URL", raising=False)

    session = _build_session(aligned_transcript="word")
    activity = AgentActivity(Agent(instructions="You are a helpful assistant."), session)

    assert activity._interruption_detection_enabled is False
    assert session.interruption_detection is NOT_GIVEN


@pytest.mark.parametrize(
    "aligned_transcript",
    [
        # inference.STT before the per-model fix, and any model that really is aligned
        pytest.param("word", id="stt-claims-alignment"),
        # the Cartesia plugin, and inference.STT for ink-2 after the fix
        pytest.param(False, id="stt-reports-no-alignment"),
    ],
)
def test_mode_vad_keeps_fast_interruption_on_either_stt(
    aligned_transcript: object,
    monkeypatch: pytest.MonkeyPatch,
    _fake_credentials: None,
) -> None:
    """`interruption.mode="vad"` is the one mitigation that holds regardless of the STT.

    `InterruptionOptions["mode"]` populates `AgentSession.interruption_detection`, which
    `_resolve_interruption_detection` short-circuits on before it ever reaches the
    hosted/dev-mode check. So it behaves the same whether the STT is the Cartesia plugin
    or the inference gateway, and stays correct after the alignment fix lands.
    """
    monkeypatch.setenv("LIVEKIT_DEV_MODE", "1")

    session = _build_session(aligned_transcript=aligned_transcript, mode="vad")
    activity = AgentActivity(Agent(instructions="You are a helpful assistant."), session)

    assert session.interruption_detection == "vad"
    assert activity._interruption_detection_enabled is False
    # the VAD barge-in path stays armed, so min_duration governs interruption speed
    assert activity._interruption_by_audio_activity_enabled is True


def test_session_vad_survives_a_turn_detecting_stt(_fake_credentials: None) -> None:
    """The session VAD must reach AgentActivity, since barge-in speed depends on it.

    `inference.STT` drops a `vad=` passed to its *own* constructor for models that
    endpoint server-side, which is easy to confuse with the session-level VAD. Only the
    latter feeds `on_vad_inference_done`, and it is untouched.
    """
    session = _build_session(aligned_transcript=False, mode="vad")
    activity = AgentActivity(Agent(instructions="You are a helpful assistant."), session)

    assert activity.vad is session.vad is not None
