# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi: `cargo run -p lk-agents-codegen -- --write`.
"""Each facade object is freed once used, closed and dropped: the core holds no strong
reference to a host object that reaches it, a cycle the collector cannot see."""

from __future__ import annotations

import gc
import weakref

import pytest

from livekit.agents._generated import (
    facades_a2a,
    facades_inference,
    facades_inference_llm,
    facades_inference_stt,
    facades_inference_tts,
    facades_llm,
    facades_llm_duplex,
    facades_llm_mcp,
    facades_llm_openai_protocol,
    facades_llm_realtime,
    facades_storage,
    facades_storage_context,
    facades_storage_session_state,
    facades_stt,
    facades_telemetry,
    facades_tokenize,
    facades_tts,
    facades_utils_aio,
    facades_voice_agent_session,
    facades_voice_background_audio,
    facades_voice_events,
    facades_voice_io,
    facades_voice_room_io,
    facades_voice_run_result,
)
from tests.fakes import until

from .facade_scenarios import SCENARIOS

FACADES: dict[str, type] = {
    "livekit.agents.a2a.Credentials": facades_a2a.Credentials,
    "livekit.agents.inference.VAD": facades_inference.VAD,
    "livekit.agents.inference._VADStream": facades_inference._VADStream,
    "livekit.agents.inference.llm.GatewayLLMStream": facades_inference_llm.GatewayLLMStream,
    "livekit.agents.inference.llm.LLM": facades_inference_llm.LLM,
    "livekit.agents.inference.stt.STT": facades_inference_stt.STT,
    "livekit.agents.inference.stt.SpeechStream": facades_inference_stt.SpeechStream,
    "livekit.agents.inference.tts.TTS": facades_inference_tts.TTS,
    "livekit.agents.llm.FallbackAdapter": facades_llm.FallbackAdapter,
    "livekit.agents.llm.FallbackLLMStream": facades_llm.FallbackLLMStream,
    "livekit.agents.llm.duplex.DuplexRealtimeAdapter": facades_llm_duplex.DuplexRealtimeAdapter,
    "livekit.agents.llm.mcp.MCPServerHTTP": facades_llm_mcp.MCPServerHTTP,
    "livekit.agents.llm.mcp.MCPServerStdio": facades_llm_mcp.MCPServerStdio,
    "livekit.agents.llm.openai_protocol.OpenAIProtocolModel": facades_llm_openai_protocol.OpenAIProtocolModel,
    "livekit.agents.llm.realtime.RealtimeModelFallbackAdapter": facades_llm_realtime.RealtimeModelFallbackAdapter,
    "livekit.agents.storage.Partition": facades_storage.Partition,
    "livekit.agents.storage.Thread": facades_storage.Thread,
    "livekit.agents.storage.Transaction": facades_storage.Transaction,
    "livekit.agents.storage.context.ContextStore": facades_storage_context.ContextStore,
    "livekit.agents.storage.context._SessionBinding": facades_storage_context._SessionBinding,
    "livekit.agents.storage.session_state.SessionState": facades_storage_session_state.SessionState,
    "livekit.agents.stt.FallbackAdapter": facades_stt.FallbackAdapter,
    "livekit.agents.stt.FallbackRecognizeStream": facades_stt.FallbackRecognizeStream,
    "livekit.agents.stt.StreamAdapter": facades_stt.StreamAdapter,
    "livekit.agents.stt.StreamAdapterWrapper": facades_stt.StreamAdapterWrapper,
    "livekit.agents.telemetry._Span": facades_telemetry._Span,
    "livekit.agents.tokenize.SentenceTokenizer": facades_tokenize.SentenceTokenizer,
    "livekit.agents.tokenize.WordTokenizer": facades_tokenize.WordTokenizer,
    "livekit.agents.tts.FallbackAdapter": facades_tts.FallbackAdapter,
    "livekit.agents.tts.StreamAdapter": facades_tts.StreamAdapter,
    "livekit.agents.utils.aio.ConnectionPool": facades_utils_aio.ConnectionPool,
    "livekit.agents.voice.agent_session.VoiceSession": facades_voice_agent_session.VoiceSession,
    "livekit.agents.voice.background_audio.PlayHandle": facades_voice_background_audio.PlayHandle,
    "livekit.agents.voice.events.RunContext": facades_voice_events.RunContext,
    "livekit.agents.voice.io.SpeechHandle": facades_voice_io.SpeechHandle,
    "livekit.agents.voice.room_io.RoomBinding": facades_voice_room_io.RoomBinding,
    "livekit.agents.voice.room_io.RoomIO": facades_voice_room_io.RoomIO,
    "livekit.agents.voice.run_result.RunStream": facades_voice_run_result.RunStream,
}


def test_every_facade_has_a_scenario() -> None:
    assert sorted(SCENARIOS) == sorted(FACADES)


@pytest.mark.parametrize("facade", sorted(FACADES))
async def test_a_closed_facade_object_is_freed(
    facade: str, caplog: pytest.LogCaptureFixture
) -> None:
    closed = await SCENARIOS[facade]()
    assert isinstance(closed, FACADES[facade])
    freed = weakref.ref(closed)
    del closed

    def collected() -> bool:
        # Expected connection failures leave captured traceback frames holding the facade.
        # Release the test's references; formatted log diagnostics remain available.
        for record in caplog.records:
            record.exc_info = None
        gc.collect()
        return freed() is None

    await until(collected, f"a closed {facade} is freed")
