"""Gemini Live API (realtime and Gemini STT) against a local peer speaking the Live JSON protocol."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from google.genai import types
from google.oauth2.credentials import Credentials

from livekit import rtc
from livekit.agents import llm, stt
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.plugins.google.beta import GeminiSTT
from livekit.plugins.google.realtime import RealtimeModel
from livekit.plugins.google.realtime.realtime_api import RealtimeSession
from tests_new.fakes import until

if TYPE_CHECKING:
    from tests_new.plugins.conftest import LivePeer

Open = Callable[..., RealtimeSession]

# ---- replies and transcripts


@pytest.mark.parametrize(
    "rejected",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="https://github.com/livekit/agents/issues/6708 reproduces: terminal rejection leaves generate_reply pending",
            ),
        ),
    ],
)
async def test_reply_requests_settle_on_output_or_terminal_rejection(
    gemini_live: LivePeer, open_session: Open, rejected: bool
) -> None:
    """A reply request settles when the server answers or rejects the turn.

    https://github.com/livekit/agents/issues/3015 https://github.com/livekit/agents/issues/6708
    """
    session = open_session()
    generations: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", generations.append)

    if rejected:
        pending = session.generate_reply()
        await gemini_live.send(
            serverContent={"turnComplete": True, "turnCompleteReason": "MALFORMED_FUNCTION_CALL"}
        )
        await until(pending.done, "terminal rejection must resolve the waiting caller", timeout=1)
        with pytest.raises(llm.RealtimeError, match="MALFORMED_FUNCTION_CALL"):
            pending.result()
        return

    for turn in range(2):
        pending = session.generate_reply()
        await gemini_live.send(
            serverContent={
                **gemini_live.AUDIO,
                "outputTranscription": {"text": f"reply {turn}"},
                "turnComplete": True,
            }
        )
        event = await pending
        assert event is generations[-1] and event.user_initiated
        assert await gemini_live.drain(event) == (f"reply {turn}", 1, [])
    assert len(generations) == 2


async def test_written_call_text_is_not_transcribed(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """Only the output transcription is the reply's text, never the model's written call syntax.

    https://github.com/livekit/agents/pull/6955
    """
    session = open_session()
    pending = session.generate_reply()

    await gemini_live.send(
        serverContent={"modelTurn": {"parts": [{"text": "call:assetGenerator{context:"}]}}
    )
    await gemini_live.send(serverContent=gemini_live.AUDIO)
    await gemini_live.send(serverContent={"outputTranscription": {"text": "Tako je!"}})
    await gemini_live.send(serverContent={"generationComplete": True, "turnComplete": True})

    assert await gemini_live.drain(await pending) == ("Tako je!", 1, [])


async def test_tool_call_is_delivered_without_its_written_call_text(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """A tool call written out as text reaches the function stream, not the transcript.

    https://github.com/livekit/agents/pull/6955
    """
    session = open_session()
    pending = session.generate_reply()

    await gemini_live.send(
        serverContent={"modelTurn": {"parts": [{"text": "call:getWeather{location:"}]}}
    )
    await gemini_live.send(
        toolCall={
            "functionCalls": [{"id": "fc-1", "name": "getWeather", "args": {"location": "Seattle"}}]
        }
    )

    assert await gemini_live.drain(await pending) == ("", 0, ["getWeather"])


async def test_text_and_audio_end_on_generation_complete(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """Audio and text end on generation_complete, which Gemini sends long before turn_complete.

    https://github.com/livekit/agents/issues/6421
    """
    session = open_session()
    pending = session.generate_reply()

    await gemini_live.send(
        serverContent={
            **gemini_live.AUDIO,
            "outputTranscription": {"text": "hello"},
            "generationComplete": True,
        }
    )
    generation = await pending
    message = await anext(aiter(generation.message_stream))

    async with asyncio.timeout(5):  # the streams end without a turn_complete
        chunks = [chunk async for chunk in message.text_stream]
        assert "".join(c if isinstance(c, str) else c.text for c in chunks) == "hello"
        assert len([frame async for frame in message.audio_stream]) == 1


async def test_content_after_generation_complete_is_dropped(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """Stray audio and text after generation_complete are dropped and the session carries on.

    https://github.com/livekit/agents/pull/6431
    """
    session = open_session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    pending = session.generate_reply()

    await gemini_live.send(serverContent={**gemini_live.AUDIO, "generationComplete": True})
    await gemini_live.send(
        serverContent={**gemini_live.AUDIO, "outputTranscription": {"text": "late"}}
    )
    await gemini_live.send(serverContent={"turnComplete": True})

    assert await gemini_live.drain(await pending) == ("", 1, [])
    assert errors == []
    assert len(gemini_live.setups) == 1


async def test_input_transcripts_are_stamped_with_their_reply_start(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """Interim and final input transcripts share the start of the reply they prompted.

    https://github.com/livekit/agents/pull/6914
    """
    session = open_session()
    transcripts: list[llm.InputTranscriptionCompleted] = []
    session.on("input_audio_transcription_completed", transcripts.append)
    before = time.time()

    await gemini_live.send(serverContent={"inputTranscription": {"text": "hello"}})
    await gemini_live.send(serverContent={"turnComplete": True})
    await until(lambda: len(transcripts) == 2, "an interim and a final transcript")

    assert [t.is_final for t in transcripts] == [False, True]
    assert transcripts[0].turn_started_at == transcripts[1].turn_started_at
    started = transcripts[0].turn_started_at
    assert started is not None and before <= started <= time.time()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="https://github.com/livekit/agents/issues/8003 reproduces: a transcript-only turn emits a false barge-in",
)
async def test_manual_activity_transcript_does_not_interrupt_the_model(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """A transcript released after a manual activity ended is no new barge-in.

    https://github.com/livekit/agents/issues/8003
    """
    session = open_session(
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
        )
    )
    starts: list[llm.InputSpeechStartedEvent] = []
    transcripts: list[llm.InputTranscriptionCompleted] = []
    session.on("input_speech_started", starts.append)
    session.on("input_audio_transcription_completed", transcripts.append)
    session.start_user_activity()
    # generate_reply closes the activity before the server releases its transcript
    session.generate_reply().cancel()
    await asyncio.sleep(0)
    starts.clear()

    await gemini_live.send(
        serverContent={"inputTranscription": {"text": "Hello?", "finished": True}}
    )
    await until(lambda: transcripts, "the transcript arrives")

    assert [event.transcript for event in transcripts] == ["Hello?"]
    assert starts == [], "the model's transcript is not evidence of a new user activity"


async def test_appended_text_then_a_reply_request_completes_one_turn(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """Syncing history sends its text once without completing the turn; the reply completes it.

    https://github.com/livekit/agents/issues/2884 https://github.com/livekit/agents/issues/3194
    https://github.com/livekit/agents/issues/3386 https://github.com/livekit/agents/issues/1636
    """
    session = open_session()
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="Hello once")

    await session.update_chat_ctx(ctx)
    pending = session.generate_reply()
    contents = await gemini_live.wait_sent("client_content", 2)
    await gemini_live.send(serverContent={**gemini_live.AUDIO, "turnComplete": True})
    await pending

    assert [c.get("turnComplete", False) for c in contents] == [False, True]
    assert json.dumps(contents).count("Hello once") == 1


async def test_fragmented_transcripts_finalize_once_when_interrupted(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """An interrupted turn keeps one complete user transcript and one reply.

    https://github.com/livekit/agents/issues/2098
    """
    session = open_session()
    finals: list[str] = []
    generations: list[llm.GenerationCreatedEvent] = []
    session.on(
        "input_audio_transcription_completed",
        lambda ev: finals.append(ev.transcript) if ev.is_final else None,
    )
    session.on("generation_created", generations.append)

    for fragment in ("  Hello", " there"):
        await gemini_live.send(serverContent={"inputTranscription": {"text": fragment}})
    await gemini_live.send(
        serverContent={**gemini_live.AUDIO, "outputTranscription": {"text": "Welcome"}}
    )
    await gemini_live.send(serverContent={"interrupted": True})
    await gemini_live.send(serverContent={"turnComplete": True})
    await gemini_live.send(serverContent={"turnComplete": True})
    await until(lambda: finals, "the user transcript is final")

    assert finals == ["Hello there"]
    assert len(generations) == 1
    assert await gemini_live.drain(generations[0]) == ("Welcome", 1, [])
    assert [(m.role, m.text_content) for m in session.chat_ctx.messages()] == [
        ("user", "Hello there"),
        ("assistant", "Welcome"),
    ]


@pytest.mark.parametrize(
    "audio",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="https://github.com/livekit/agents/issues/3540 reproduces: text-only setup carries speech_config",
            ),
        ),
    ],
)
async def test_text_only_setup_does_not_request_a_voice(
    gemini_live: LivePeer, open_session: Open, audio: bool
) -> None:
    """A text-only session asks for no voice and gets a text reply.

    https://github.com/livekit/agents/issues/3540
    """
    session = open_session(modalities=[types.Modality.AUDIO if audio else types.Modality.TEXT])
    pending = session.generate_reply()
    content = gemini_live.AUDIO if audio else {"modelTurn": {"parts": [{"text": "Text answer"}]}}
    await gemini_live.send(serverContent={**content, "turnComplete": True})

    assert await gemini_live.drain(await pending) == (
        ("", 1, []) if audio else ("Text answer", 0, [])
    )
    assert ("speechConfig" in gemini_live.setups[0]["generationConfig"]) == audio


@pytest.mark.parametrize(
    "vertex",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="https://github.com/livekit/agents/issues/7785 reproduces: Vertex updates append a model turn",
            ),
        ),
    ],
)
async def test_instruction_update_uses_the_backend_instruction_role_without_reconnect(
    gemini_live: LivePeer, open_session: Open, vertex: bool
) -> None:
    """New instructions go over the live socket in the role the backend reads as instructions.

    https://github.com/livekit/agents/issues/7785
    """
    session = open_session(vertexai=vertex)
    await gemini_live.ready(session)

    await session.update_instructions("Use the new policy")
    [content] = await gemini_live.wait_sent("client_content")

    assert content["turns"] == [
        {"role": "system" if vertex else "model", "parts": [{"text": "Use the new policy"}]}
    ]
    assert not content.get("turnComplete")
    assert len(gemini_live.setups) == 1


@pytest.mark.parametrize(
    "handle",
    [
        "saved-handle",
        pytest.param(
            None,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="https://github.com/livekit/agents/issues/5102 reproduces: initial setup sends an empty resumption config",
            ),
        ),
    ],
)
async def test_initial_setup_omits_null_resumption_but_keeps_saved_handles(
    gemini_live: LivePeer, open_session: Open, handle: str | None
) -> None:
    """Only a real resumption handle reaches the setup. https://github.com/livekit/agents/issues/5102"""
    open_session(
        model="gemini-2.5-flash-native-audio-latest",
        session_resumption=types.SessionResumptionConfig(handle=handle),
    )
    await until(lambda: gemini_live.setups, "the plugin connects")

    if handle is None:
        assert "sessionResumption" not in gemini_live.setups[0]
    else:
        assert gemini_live.setups[0]["sessionResumption"]["handle"] == handle


@pytest.mark.parametrize(
    "model",
    [
        "gemini-2.5-flash-native-audio-preview-12-2025",
        pytest.param(
            "gemini-3.1-flash-live-preview",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="https://github.com/livekit/agents/pull/7673 reproduces: 3.x reply instructions use an unwrapped model turn",
            ),
        ),
        pytest.param(
            "gemini-3.8-live",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="https://github.com/livekit/agents/pull/7673 reproduces: 3.x reply instructions use an unwrapped model turn",
            ),
        ),
    ],
)
async def test_reply_instructions_are_wrapped_for_models_without_placeholders(
    gemini_live: LivePeer, open_session: Open, model: str
) -> None:
    """Reply instructions cannot become assistant prefill. https://github.com/livekit/agents/pull/7673"""
    session = open_session(model=model)

    session.generate_reply(instructions="Greet the caller")
    [content] = await gemini_live.wait_sent("client_content")

    turns = [(turn["role"], turn["parts"][0]["text"]) for turn in content["turns"]]
    if "2.5" in model:
        assert turns == [("model", "Greet the caller"), ("user", ".")]
    else:
        assert turns == [("user", "<instructions>\nGreet the caller\n</instructions>")]


async def test_usage_reaches_metrics_with_tool_prompts_and_audio_counts(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """A usage-only message reaches metrics: tool prompts count as input, null counts are unknown.

    https://github.com/livekit/agents/pull/5404 https://github.com/livekit/agents/issues/2426
    """
    session = open_session()
    metrics: list[object] = []
    session.on("metrics_collected", metrics.append)

    await gemini_live.send(serverContent={**gemini_live.AUDIO, "turnComplete": True})
    await gemini_live.send(
        usageMetadata={
            "promptTokenCount": 20,
            "toolUsePromptTokenCount": 3,
            "responseTokenCount": 10,
            "thoughtsTokenCount": 2,
            "promptTokensDetails": [
                {"modality": "AUDIO", "tokenCount": 11},
                {"modality": "TEXT"},
            ],
            "toolUsePromptTokensDetails": [{"modality": "AUDIO", "tokenCount": 2}],
            "responseTokensDetails": [{"modality": "AUDIO", "tokenCount": 7}],
        }
    )
    await until(lambda: len(metrics) == 2, "usage reaches metrics")

    [_connection, record] = metrics
    assert isinstance(record, RealtimeModelMetrics)
    assert (record.tokens.input_tokens, record.tokens.output_tokens) == (23, 12)
    assert (record.tokens.input_audio_tokens, record.tokens.output_audio_tokens) == (13, 7)
    assert record.tokens.input_text_tokens is None


@pytest.mark.parametrize("vertex", [False, True])
@pytest.mark.parametrize("with_input_transcript", [False, True])
async def test_ttft_is_unavailable_without_a_response_start_event(
    gemini_live: LivePeer,
    open_session: Open,
    vertex: bool,
    with_input_transcript: bool,
) -> None:
    """Gemini marks no response start, so a turn reports no time to first token.

    https://github.com/livekit/agents/pull/7719
    """
    session = open_session(model="gemini-3.8-live", vertexai=vertex)
    metrics: list[object] = []
    session.on("metrics_collected", metrics.append)

    if with_input_transcript:
        await gemini_live.send(serverContent={"inputTranscription": {"text": "Hello"}})
    await gemini_live.send(serverContent=gemini_live.AUDIO)
    # Vertex names the response count after its candidates
    output_count = "candidatesTokenCount" if vertex else "responseTokenCount"
    await gemini_live.send(usageMetadata={"promptTokenCount": 10, output_count: 5})
    await until(lambda: len(metrics) == 2, "usage reaches metrics")

    [_connection, record] = metrics
    assert isinstance(record, RealtimeModelMetrics)
    assert record.ttft is None
    assert (record.tokens.input_tokens, record.tokens.output_tokens) == (10, 5)


@pytest.mark.parametrize("explicit", [False, True])
async def test_vertex_sessions_authenticate_with_workload_credentials(
    gemini_live: LivePeer, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    """Ambient or explicit workload credentials authenticate a Vertex session without key files.

    https://github.com/livekit/agents/issues/1428 https://github.com/livekit/agents/issues/2387
    """
    import google.auth

    for name in ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    ambient = Credentials(token="ambient-token")  # type: ignore[no-untyped-call]  # google-auth is untyped
    monkeypatch.setattr(google.auth, "default", lambda *a, **kw: (ambient, "workload-project"))
    explicit_token = Credentials(token="explicit-token")  # type: ignore[no-untyped-call]  # google-auth is untyped
    credentials = explicit_token if explicit else None
    session = RealtimeModel(
        vertexai=True, credentials=credentials, http_options=gemini_live.http_options
    ).session()
    try:
        pending = session.generate_reply()
        await gemini_live.send(
            serverContent={
                **gemini_live.AUDIO,
                "outputTranscription": {"text": "Authenticated"},
                "turnComplete": True,
            }
        )
        assert await gemini_live.drain(await pending) == ("Authenticated", 1, [])
    finally:
        await session.aclose()

    token = "explicit-token" if explicit else "ambient-token"
    assert gemini_live.headers[0]["Authorization"] == f"Bearer {token}"
    assert gemini_live.setups[0]["model"].startswith("projects/workload-project/")


# ---- Gemini STT


async def test_gemini_stt_emits_the_finalized_transcript_on_arrival(gemini_live: LivePeer) -> None:
    """Interims resend the full hypothesis; input_transcription is the one final.

    https://github.com/livekit/agents/pull/6991
    """
    stream = GeminiSTT(api_key="test", http_options=gemini_live.http_options).stream()
    stream.push_frame(rtc.AudioFrame(bytes(320), 16000, 1, 160))
    stream.end_input()

    await gemini_live.send(serverContent={"interimInputTranscription": {"text": "Greetings."}})
    await gemini_live.send(
        serverContent={"interimInputTranscription": {"text": "Greetings, welcome"}}
    )
    await gemini_live.send(
        serverContent={"inputTranscription": {"text": "Greetings, welcome to the age of AI."}}
    )
    await gemini_live.send(serverContent={"generationComplete": True, "turnComplete": True})
    async with stream:
        events = [(e.type, e.alternatives[0].text) async for e in stream]

    assert events == [
        (stt.SpeechEventType.INTERIM_TRANSCRIPT, "Greetings."),
        (stt.SpeechEventType.INTERIM_TRANSCRIPT, "Greetings, welcome"),
        (stt.SpeechEventType.FINAL_TRANSCRIPT, "Greetings, welcome to the age of AI."),
    ]
