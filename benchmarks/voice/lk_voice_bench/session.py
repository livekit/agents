"""Run the agent under test as an in-process ``AgentSession`` wired to a benchmark."""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentSession,
    TurnHandlingOptions,
    inference,
    llm,
)
from livekit.agents.llm import ToolError, function_tool
from livekit.agents.utils import http_context
from livekit.agents.voice.transcription import TranscriptSynchronizer

from .audio_io import CallerAudioInput, SpeakerOutput, SpokenTextOutput
from .config import AgentConfig

logger = logging.getLogger("lk_voice_bench")

ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]
"""``(tool_name, arguments) -> result``; raise ``ToolError`` for an error result."""


@dataclass
class SessionLog:
    """What happened on the agent side, for analysis next to the benchmark's own logs."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, kind: str, **data: Any) -> None:
        self.events.append({"t": time.time(), "kind": kind, **data})


def _proxy_tool(schema: dict[str, Any], executor: ToolExecutor, log: SessionLog) -> llm.Tool:
    name = schema["name"]

    async def _call(raw_arguments: dict[str, Any]) -> str:
        log.add("tool_call", name=name, arguments=raw_arguments)
        try:
            result = await executor(name, raw_arguments)
        except ToolError as e:
            log.add("tool_result", name=name, error=True, result=e.message)
            raise
        log.add("tool_result", name=name, error=False, result=result)
        return result

    return function_tool(
        _call,
        raw_schema={
            "name": name,
            "description": schema.get("description", ""),
            "parameters": schema.get("parameters") or {"type": "object", "properties": {}},
        },
    )


def build_session(config: AgentConfig) -> AgentSession[None]:
    tts_extra: dict[str, Any] = {}
    if config.tts_word_timestamps and config.tts.startswith("cartesia/"):
        tts_extra["add_timestamps"] = True

    return AgentSession(
        stt=inference.STT(config.stt, language=config.stt_language),
        llm=inference.LLM(
            config.llm,
            reasoning=config.llm_reasoning if config.llm_reasoning else NOT_GIVEN,  # type: ignore[arg-type]
        ),
        tts=inference.TTS(config.tts, voice=config.tts_voice, extra_kwargs=tts_extra),
        turn_handling=TurnHandlingOptions(
            interruption={
                "resume_false_interruption": config.resume_false_interruption,
                "false_interruption_timeout": config.false_interruption_timeout,
            },
            preemptive_generation={"enabled": config.preemptive_generation, "max_retries": 3},
        ),
        max_tool_steps=config.max_tool_steps,
        use_tts_aligned_transcript=config.tts_word_timestamps,
    )


class BenchAgent:
    """The agent under test, driven by a benchmark through PCM in and PCM out.

    All methods must run on one event loop, which also runs the session.
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        instructions: str,
        tools: list[dict[str, Any]],
        tool_executor: ToolExecutor,
        greeting: str | None = None,
        output_sample_rate: int = 8000,
        external_clock: bool = False,
    ) -> None:
        """
        Args:
            instructions: The benchmark's system prompt, used verbatim.
            tools: Function schemas (``name``, ``description``, ``parameters``).
            tool_executor: Runs a tool call against the benchmark's environment.
            greeting: An opening line the benchmark already "said" for the agent; it goes in the
                chat history as the agent's first message, and is not spoken.
            output_sample_rate: Sample rate of the PCM collected from the virtual speaker.
            external_clock: The benchmark plays the speaker out itself, with
                ``speaker.play_frame()`` every 20 ms of its own time, instead of the speaker
                playing on its own real-time clock.
        """
        self.config = config
        self._external_clock = external_clock
        self.log = SessionLog()
        self.mic = CallerAudioInput()
        self.speaker = SpeakerOutput(sample_rate=output_sample_rate)
        self.spoken_text = SpokenTextOutput(self.speaker)
        self._sync = TranscriptSynchronizer(
            next_in_chain_audio=self.speaker, next_in_chain_text=self.spoken_text
        )

        chat_ctx = llm.ChatContext()
        if greeting:
            chat_ctx.add_message(role="assistant", content=greeting)
        self.agent = Agent(
            instructions=instructions,
            tools=[_proxy_tool(s, tool_executor, self.log) for s in tools],
            chat_ctx=chat_ctx,
        )
        self.session = build_session(config)
        self._wire_logging()

    def _wire_logging(self) -> None:
        s = self.session

        @s.on("user_input_transcribed")
        def _on_user(ev: Any) -> None:
            if ev.is_final:
                self.log.add("user_transcript", text=ev.transcript)

        @s.on("conversation_item_added")
        def _on_item(ev: Any) -> None:
            item = ev.item
            if getattr(item, "type", None) == "message":
                self.log.add(
                    "message",
                    role=item.role,
                    text=item.text_content,
                    interrupted=item.interrupted,
                    metrics=dict(item.metrics) if getattr(item, "metrics", None) else None,
                )

        @s.on("agent_state_changed")
        def _on_agent_state(ev: Any) -> None:
            self.log.add("agent_state", old=ev.old_state, new=ev.new_state)

        @s.on("user_state_changed")
        def _on_user_state(ev: Any) -> None:
            self.log.add("user_state", old=ev.old_state, new=ev.new_state)

        @s.on("error")
        def _on_error(ev: Any) -> None:
            self.log.add("error", error=repr(ev.error), source=type(ev.source).__name__)
            logger.warning("agent session error: %r", ev.error)

    async def start(self) -> None:
        # outside a job there is no shared http session; bind one to the context the session's
        # tasks are created in (they inherit it), and close it in aclose()
        self._own_http_session = http_context._ContextVar.get(None) is None
        if self._own_http_session:
            self._http_session_factory = http_context._new_session_ctx()
        self.session.input.audio = self.mic
        self.session.output.audio = self._sync.audio_output
        self.session.output.transcription = self._sync.text_output
        if not self._external_clock:
            self.speaker.start()
        await self.session.start(agent=self.agent, record=False)
        self.log.add("started", config=self.config.to_dict())

    def push_caller_audio(self, pcm: bytes, sample_rate: int) -> None:
        self.mic.push_pcm(pcm, sample_rate)

    async def aclose(self) -> None:
        try:
            self.log.add("usage", usage=dataclasses.asdict(self.session.usage))
        except Exception as e:  # usage is diagnostics only
            self.log.add("usage", error=repr(e))
        self.mic.close()
        await self.session.aclose()
        await self._sync.aclose()
        await self.speaker.aclose()
        if self._own_http_session:
            await self._http_session_factory().close()
        self.log.add("closed")

    def dump_log(self, path: str) -> None:
        with open(path, "w") as f:
            for ev in self.log.events:
                f.write(json.dumps(ev, default=str) + "\n")
