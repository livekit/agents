"""tau2-bench voice (tau-Voice) adapter: a LiveKit Agents ``AgentSession`` as the agent under test.

Registered in tau2 as the ``livekit_agents`` audio-native provider by
``benchmarks/voice/tau2/register-livekit-agents.patch``; ``--audio-native-model`` names an agent
preset from ``lk_voice_bench.config``.

The session runs in-process on the adapter's background event loop, on tau2's clock:

- each tick (200 ms by default, paced in wall-clock time) the caller's audio is pushed into the
  session's microphone in 20 ms frames, and the session's virtual speaker plays one 20 ms frame
  per slot, so the audio tau2 records is exactly the audio the session believes it played;
- tool calls made by the session's LLM are handed to tau2 as ``TickResult.tool_calls`` and the
  session's tool call waits until tau2's result comes back at the start of a later tick;
- the caller simulator reads the agent's words from ``proportional_transcript``. It is fed from
  LiveKit's transcript synchronizer (word timestamps from the TTS), so an interrupted turn is
  cut at the word the speaker actually reached.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioEncoding, AudioFormat
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript

from livekit.agents.llm import ToolError

from .audio_io import FRAME_MS
from .codec import pcm16_to_ulaw, ulaw_to_pcm16
from .config import get_preset
from .session import BenchAgent

DEFAULT_OPENER = "Hi! How can I help you today?"  # tau2's create_initial_message default
TOOL_RESULT_TIMEOUT = 120.0
LOG_DIR_ENV = "LK_BENCH_LOG_DIR"
"""If set, each simulation's agent-side event log is written there as JSONL."""


@dataclass
class _TickEvent:
    """Marks provider activity in a tick, for tau2's inactivity check."""

    type: str
    data: Any = None


class LiveKitAgentsAdapter(DiscreteTimeAdapter):
    def __init__(
        self,
        tick_duration_ms: float,
        send_audio_instant: bool = False,
        model: str | None = None,
        audio_format: AudioFormat | None = None,
    ) -> None:
        # tau2 derives it from seconds (0.2 * 1000), so it arrives as a float
        tick_ms = round(tick_duration_ms)
        if abs(tick_ms - tick_duration_ms) > 1e-6 or tick_ms % FRAME_MS:
            raise ValueError(f"tick duration must be a multiple of {FRAME_MS} ms")
        super().__init__(
            tick_duration_ms=tick_ms,
            audio_format=audio_format or TELEPHONY_AUDIO_FORMAT,
            send_audio_instant=send_audio_instant,
        )
        if self.audio_format.encoding not in (AudioEncoding.ULAW, AudioEncoding.PCM_S16LE):
            raise ValueError(f"unsupported audio format {self.audio_format}")
        if self.audio_format.channels != 1:
            raise ValueError("only mono audio is supported")
        # ticks follow the wall clock, which the AgentSession runs on
        self.realtime_pacing = True
        self.config = get_preset(model or "baseline-gemma4")
        self._opener = DEFAULT_OPENER
        self._bg = BackgroundAsyncLoop()
        self._agent: BenchAgent | None = None
        self._outgoing_calls: list[ToolCall] = []
        self._awaiting_results: dict[str, asyncio.Future[tuple[str, bool]]] = {}
        self._user_vad_events: list[str] = []
        self._sim_id = uuid.uuid4().hex[:12]

    # -- connection ---------------------------------------------------------------------------

    def set_opening_message(self, content: str | None) -> None:
        """The line tau2 has the agent open with (not synthesized); kept in its chat history."""
        if content:
            self._opener = content

    @property
    def is_connected(self) -> bool:
        return self._agent is not None

    def connect(
        self,
        system_prompt: str,
        tools: list[Tool],
        vad_config: Any,
        modality: str = "audio",
    ) -> None:
        if modality != "audio":
            raise ValueError(f"livekit_agents supports modality 'audio' only, not {modality!r}")
        self._bg.start()
        schemas = [tool.openai_schema["function"] for tool in tools]
        self._bg.run_coroutine(self._async_connect(system_prompt, schemas), timeout=60.0)
        logger.info(
            f"livekit_agents connected: preset={self.config.name} sim={self._sim_id} "
            f"tools={len(schemas)}"
        )

    async def _async_connect(self, system_prompt: str, schemas: list[dict[str, Any]]) -> None:
        agent = BenchAgent(
            self.config,
            instructions=system_prompt,
            tools=schemas,
            tool_executor=self._execute_tool,
            greeting=self._opener,
            output_sample_rate=self.audio_format.sample_rate,
            external_clock=True,
        )

        @agent.session.on("user_state_changed")
        def _on_user_state(ev: Any) -> None:
            if ev.new_state == "speaking":
                self._user_vad_events.append("speech_started")
            elif ev.old_state == "speaking":
                self._user_vad_events.append("speech_stopped")

        await agent.start()
        self._agent = agent

    def disconnect(self) -> None:
        if self._agent is None:
            return
        try:
            self._bg.run_coroutine(self._async_disconnect(), timeout=30.0)
        except Exception as e:
            logger.warning(f"livekit_agents disconnect failed: {e!r}")
        finally:
            self._agent = None
            self._bg.stop()
            self.clear_buffers()

    async def _async_disconnect(self) -> None:
        assert self._agent is not None
        for fut in self._awaiting_results.values():
            if not fut.done():
                fut.cancel()
        await self._agent.aclose()
        if log_dir := os.environ.get(LOG_DIR_ENV):
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self._agent.dump_log(str(Path(log_dir) / f"{stamp}_{self._sim_id}.jsonl"))

    # -- tools --------------------------------------------------------------------------------

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        call_id = f"call_{uuid.uuid4().hex[:16]}"
        fut: asyncio.Future[tuple[str, bool]] = asyncio.get_running_loop().create_future()
        self._awaiting_results[call_id] = fut
        self._outgoing_calls.append(
            ToolCall(id=call_id, name=name, arguments=arguments, requestor="assistant")
        )
        try:
            result, is_error = await asyncio.wait_for(fut, timeout=TOOL_RESULT_TIMEOUT)
        finally:
            self._awaiting_results.pop(call_id, None)
        if is_error:
            raise ToolError(result)
        return result

    async def _flush_pending_tool_results(self) -> None:
        pending, self._pending_tool_results = self._pending_tool_results, []
        for call_id, result, _request_response, is_error in pending:
            # the session always replies after a tool result, so request_response is moot
            fut = self._awaiting_results.get(call_id)
            if fut is None or fut.done():
                logger.warning(f"livekit_agents: result for unknown tool call {call_id}")
                continue
            fut.set_result((result, is_error))

    # -- ticks --------------------------------------------------------------------------------

    def run_tick(self, user_audio: bytes, tick_number: int | None = None) -> TickResult:
        if self._agent is None:
            raise RuntimeError("livekit_agents adapter is not connected")
        if tick_number is None:
            tick_number = self._tick_count + 1
        self._tick_count = tick_number
        return self._bg.run_coroutine(
            self._async_run_tick(user_audio, tick_number),
            timeout=self.tick_duration_ms / 1000 + 30.0,
        )

    async def _async_run_tick(self, user_audio: bytes, tick_number: int) -> TickResult:
        result = await super()._async_run_tick(user_audio, tick_number)
        # tau2 only voices text for utterances with audio in this tick; the synchronizer can
        # release an utterance's last words just after its last frame, so pass those on here
        late = []
        playing = set(result.item_ids)
        for ut in self._utterance_transcripts.values():
            if ut.item_id in playing or ut.transcript_chars_shown >= len(ut.transcript_received):
                continue
            late.append(ut.transcript_received[ut.transcript_chars_shown :])
            ut.transcript_chars_shown = len(ut.transcript_received)
        if late:
            result.proportional_transcript = " ".join(
                t for t in (result.proportional_transcript, *late) if t
            )
        return result

    def _decode(self, data: bytes) -> bytes:
        if self.audio_format.encoding == AudioEncoding.ULAW:
            return ulaw_to_pcm16(data)
        return data

    def _encode(self, pcm: bytes) -> bytes:
        if self.audio_format.encoding == AudioEncoding.ULAW:
            return pcm16_to_ulaw(pcm)
        return pcm

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        agent = self._agent
        assert agent is not None
        loop = asyncio.get_running_loop()
        rate = self.audio_format.sample_rate
        frame_bytes = rate * FRAME_MS // 1000 * 2
        pcm = self._decode(user_audio)
        frames = [pcm[i : i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]
        slots = self.tick_duration_ms // FRAME_MS

        # one 20 ms slot at a time: the caller's frame goes in, the speaker plays a frame out
        for i in range(slots):
            delay = tick_start + i * FRAME_MS / 1000 - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            if i < len(frames):
                last = i == slots - 1
                for frame in frames[i:] if last else frames[i : i + 1]:
                    agent.push_caller_audio(frame, rate)
            agent.speaker.play_frame()

        self._collect(agent, result)

    def _collect(self, agent: BenchAgent, result: TickResult) -> None:
        if self._outgoing_calls:
            result.tool_calls.extend(self._outgoing_calls)
            result.events.append(_TickEvent("tool_calls", [c.name for c in self._outgoing_calls]))
            self._outgoing_calls = []

        for chunk in agent.speaker.drain_played():
            data = self._encode(chunk.pcm)
            result.agent_audio_chunks.append((data, chunk.segment_id))
            self._transcript_for(chunk.segment_id).add_audio(len(data))
        if result.agent_audio_chunks:
            result.events.append(_TickEvent("audio"))

        for spoken in agent.spoken_text.drain():
            if spoken.segment_id is None:
                continue
            self._transcript_for(spoken.segment_id).add_transcript(spoken.text)

        interruptions = agent.speaker.drain_interruptions()
        if interruptions:
            result.was_truncated = True
            # dropped bytes are int16 PCM; count them in the external format
            scale = 2 if self.audio_format.encoding == AudioEncoding.ULAW else 1
            result.truncated_audio_bytes += sum(i.dropped_bytes // scale for i in interruptions)
            result.vad_events.append("interrupted")
            result.events.append(_TickEvent("interrupted", [i.segment_id for i in interruptions]))

        if self._user_vad_events:
            result.vad_events.extend(self._user_vad_events)
            result.events.append(_TickEvent("user_vad", list(self._user_vad_events)))
            self._user_vad_events = []

    def _transcript_for(self, segment_id: str) -> UtteranceTranscript:
        ut = self._utterance_transcripts.get(segment_id)
        if ut is None:
            ut = self._utterance_transcripts[segment_id] = UtteranceTranscript(item_id=segment_id)
        return ut
