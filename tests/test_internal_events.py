from __future__ import annotations

import base64
import json
import time
from typing import get_args
from unittest.mock import MagicMock

from livekit import rtc
from livekit.agents import Agent, llm
from livekit.agents.llm import (
    ChatChunk,
    FunctionCall,
    FunctionCallOutput,
    GenerationCreatedEvent,
    InputSpeechStartedEvent,
    InputSpeechStoppedEvent,
    InputTranscriptionCompleted,
    LLMOutputEvent,
)
from livekit.agents.tts import SynthesizedAudio
from livekit.agents.types import FlushSentinel, TimedString
from livekit.agents.vad import VADEvent, VADEventType
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    RunEvent,
    TimedInternalEvent,
    UserStateChangedEvent,
    _internal_event_serializer,
    _serialize_audio_frame,
)
from livekit.agents.voice.run_result import (
    AgentHandoffEvent,
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
)


def assert_json_serializable(data: dict | None) -> None:
    """Assert that data can be serialized to JSON and back."""
    if data is None:
        return
    json_str = json.dumps(data)
    roundtrip = json.loads(json_str)
    assert roundtrip == data


class TestSerializeAudioFrame:
    def test_serializes_audio_frame_correctly(self):
        audio_data = b"\x00\x01\x02\x03"
        frame = rtc.AudioFrame(
            data=audio_data,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2,
        )

        result = _serialize_audio_frame(frame)

        assert result["sample_rate"] == 48000
        assert result["num_channels"] == 1
        assert result["samples_per_channel"] == 2
        assert result["data"] == base64.b64encode(audio_data).decode("utf-8")
        assert_json_serializable(result)


class TestRunEventHandling:
    def test_run_event_types_are_correct(self):
        event_types = get_args(RunEvent)
        assert ChatMessageEvent in event_types
        assert FunctionCallEvent in event_types
        assert FunctionCallOutputEvent in event_types
        assert AgentHandoffEvent in event_types

    def test_isinstance_with_get_args_works(self):
        msg = llm.ChatMessage(role="user", content=["test"])
        event = ChatMessageEvent(item=msg)

        assert isinstance(event, get_args(RunEvent))

    def test_chat_message_event_serialization(self):
        msg = llm.ChatMessage(role="user", content=["Hello"])
        event = ChatMessageEvent(item=msg)

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == "message"
        assert "item" in result
        assert result["item"]["role"] == "user"
        assert_json_serializable(result)

    def test_function_call_event_serialization(self):
        fnc = FunctionCall(call_id="call_123", name="get_weather", arguments='{"location": "NYC"}')
        event = FunctionCallEvent(item=fnc)

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == "function_call"
        assert result["item"]["name"] == "get_weather"
        assert result["item"]["call_id"] == "call_123"
        assert_json_serializable(result)

    def test_function_call_output_event_serialization(self):
        output = FunctionCallOutput(call_id="123", output="Sunny", is_error=False)
        event = FunctionCallOutputEvent(item=output)

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == "function_call_output"
        assert result["item"]["output"] == "Sunny"
        assert_json_serializable(result)


class TestAgentHandoffEventSerialization:
    def test_serializes_agent_ids_not_objects(self):
        old_agent = Agent(instructions="old")
        new_agent = Agent(instructions="new")
        handoff_item = llm.AgentHandoff(old_agent_id=old_agent.id, new_agent_id=new_agent.id)

        event = AgentHandoffEvent(
            item=handoff_item,
            old_agent=old_agent,
            new_agent=new_agent,
        )

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == "agent_handoff"
        assert result["old_agent"] == old_agent.id
        assert result["new_agent"] == new_agent.id
        assert "item" in result
        assert_json_serializable(result)

    def test_handles_none_old_agent(self):
        new_agent = Agent(instructions="new")
        handoff_item = llm.AgentHandoff(old_agent_id=None, new_agent_id=new_agent.id)

        event = AgentHandoffEvent(
            item=handoff_item,
            old_agent=None,
            new_agent=new_agent,
        )

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["old_agent"] is None
        assert result["new_agent"] == new_agent.id
        assert_json_serializable(result)


class TestVADEventSerialization:
    def test_inference_done_events_are_filtered(self):
        event = VADEvent(
            type=VADEventType.INFERENCE_DONE,
            samples_index=0,
            timestamp=0.0,
            speech_duration=0.0,
            silence_duration=0.0,
            probability=0.5,
        )

        result = _internal_event_serializer(event)

        assert result is None
        assert_json_serializable(result)

    def test_start_of_speech_is_serialized(self):
        event = VADEvent(
            type=VADEventType.START_OF_SPEECH,
            samples_index=100,
            timestamp=1.0,
            speech_duration=0.1,
            silence_duration=0.0,
            probability=0.9,
        )

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == VADEventType.START_OF_SPEECH
        assert result["frames"] == []
        assert_json_serializable(result)

    def test_end_of_speech_is_serialized(self):
        event = VADEvent(
            type=VADEventType.END_OF_SPEECH,
            samples_index=200,
            timestamp=3.0,
            speech_duration=2.0,
            silence_duration=0.5,
            probability=0.1,
        )

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == VADEventType.END_OF_SPEECH
        assert_json_serializable(result)


class TestGenerationCreatedEventSerialization:
    def test_strips_non_serializable_streams(self):
        event = GenerationCreatedEvent(
            message_stream=MagicMock(),
            function_stream=MagicMock(),
            user_initiated=True,
            response_id="resp_123",
        )

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["message_stream"] is None
        assert result["function_stream"] is None
        assert result["user_initiated"] is True
        assert result["response_id"] == "resp_123"
        assert result["type"] == "generation_created"
        assert_json_serializable(result)


class TestLLMOutputEventSerialization:
    def test_string_data(self):
        event = LLMOutputEvent(type="llm_str_output", data="Hello world")

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == "llm_str_output"
        assert result["data"] == "Hello world"
        assert_json_serializable(result)

    def test_timed_string_data(self):
        timed = TimedString("Hello", start_time=0.0, end_time=1.0, confidence=0.95)
        event = LLMOutputEvent(type="llm_timed_string_output", data=timed)

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == "llm_timed_string_output"
        assert result["data"]["text"] == "Hello"
        assert result["data"]["start_time"] == 0.0
        assert result["data"]["end_time"] == 1.0
        assert result["data"]["confidence"] == 0.95
        assert_json_serializable(result)

    def test_chat_chunk_data(self):
        chunk = ChatChunk(id="chunk_1", delta=None)
        event = LLMOutputEvent(type="llm_chunk_output", data=chunk)

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == "llm_chunk_output"
        assert result["data"]["id"] == "chunk_1"
        assert_json_serializable(result)

    def test_audio_frame_data(self):
        audio_data = b"\x00\x01\x02\x03"
        frame = rtc.AudioFrame(
            data=audio_data,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2,
        )
        event = LLMOutputEvent(type="realtime_audio_output", data=frame)

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == "realtime_audio_output"
        assert result["data"]["sample_rate"] == 48000
        assert result["data"]["data"] == base64.b64encode(audio_data).decode("utf-8")
        assert_json_serializable(result)


class TestSynthesizedAudioSerialization:
    def test_serializes_with_audio_frame(self):
        audio_data = b"\x00\x01\x02\x03"
        frame = rtc.AudioFrame(
            data=audio_data,
            sample_rate=24000,
            num_channels=1,
            samples_per_channel=2,
        )
        event = SynthesizedAudio(
            frame=frame,
            request_id="req_123",
            segment_id="seg_1",
            delta_text="Hello",
        )

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == "synthesized_audio"
        assert result["request_id"] == "req_123"
        assert result["frame"]["sample_rate"] == 24000
        assert_json_serializable(result)


class TestBaseModelEventSerialization:
    def test_agent_state_changed_event(self):
        event = AgentStateChangedEvent(old_state="idle", new_state="speaking")

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == "agent_state_changed"
        assert result["old_state"] == "idle"
        assert result["new_state"] == "speaking"
        assert_json_serializable(result)

    def test_user_state_changed_event(self):
        event = UserStateChangedEvent(old_state="listening", new_state="speaking")

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == "user_state_changed"
        assert result["old_state"] == "listening"
        assert result["new_state"] == "speaking"
        assert_json_serializable(result)


class TestDataclassEventSerialization:
    def test_flush_sentinel(self):
        event = FlushSentinel()

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == "flush_sentinel"
        assert_json_serializable(result)

    def test_input_speech_started(self):
        event = InputSpeechStartedEvent()

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == "input_speech_started"
        assert_json_serializable(result)

    def test_input_speech_stopped(self):
        event = InputSpeechStoppedEvent(user_transcription_enabled=True)

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == "input_speech_stopped"
        assert result["user_transcription_enabled"] is True
        assert_json_serializable(result)

    def test_input_transcription_completed(self):
        event = InputTranscriptionCompleted(
            item_id="item_123", transcript="Hello world", is_final=True
        )

        result = _internal_event_serializer(event)

        assert result is not None
        assert result["type"] == "input_transcription_completed"
        assert result["transcript"] == "Hello world"
        assert result["is_final"] is True
        assert result["item_id"] == "item_123"
        assert_json_serializable(result)


class TestTimedInternalEvent:
    def test_adds_timestamp(self):
        event = UserStateChangedEvent(old_state="listening", new_state="speaking")
        before = time.time()

        timed_event = TimedInternalEvent(event=event)

        after = time.time()
        assert before <= timed_event.created_at <= after

    def test_serialization_uses_alias(self):
        event = UserStateChangedEvent(old_state="listening", new_state="speaking")
        timed_event = TimedInternalEvent(event=event)

        serialized = timed_event.model_dump(mode="json", by_alias=True)

        assert "__created_at" in serialized
        assert "created_at" not in serialized
        assert_json_serializable(serialized)

    def test_event_is_serialized(self):
        event = UserStateChangedEvent(old_state="listening", new_state="speaking")
        timed_event = TimedInternalEvent(event=event)

        serialized = timed_event.model_dump(mode="json", by_alias=True)

        assert serialized["event"] is not None
        assert serialized["event"]["type"] == "user_state_changed"
        assert_json_serializable(serialized)

    def test_filtered_event_becomes_none(self):
        event = VADEvent(
            type=VADEventType.INFERENCE_DONE,
            samples_index=0,
            timestamp=0.0,
            speech_duration=0.0,
            silence_duration=0.0,
            probability=0.5,
        )
        timed_event = TimedInternalEvent(event=event)

        serialized = timed_event.model_dump(mode="json", by_alias=True)

        assert serialized["event"] is None
        assert_json_serializable(serialized)
