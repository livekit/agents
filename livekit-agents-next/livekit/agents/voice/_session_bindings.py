"""Python SDK capabilities and public-to-native voice-session argument adaptation."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import weakref
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, cast

from livekit import rtc

from .. import _bindings, _core, inference, llm, stt, tts
from .._core_view import (
    TurnDetector,
    tool_specs,
)
from .._generated.keyterms import STTContextOptions
from .._generated.session_options import SessionKwargs
from .._generated.session_owner import ProviderBindings
from .._generated.vad import VAD
from .._generated.voice_session import UpdateAgentArgs
from ..inference.llm import llm_from_model_string
from ..llm import (
    LLM,
    ChatContext,
    RealtimeModel,
    function_tool,
)
from ..log import logger
from ..telemetry.traces import capture_content_enabled
from ..types import (
    NOT_GIVEN,
    NotGivenOr,
)
from ..utils.misc import is_dev_mode, is_given, is_hosted, shortuuid
from . import io
from ._session_callbacks import _DynamicInstructions, _hook_name, _origin, _SessionNodes
from ._session_types import RecordingOptions, SessionConnectOptions, VoiceActivityVideoSampler
from .agent import (
    Agent,
    _BindingContextVar,
    _VoiceBinding,
)
from .events import (
    Context_T,
)
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from .._core import VoiceAgent as _NativeAgent
    from .._generated.session_options import SessionLegs, UpdateKwargs
    from ..storage.resolve import Thread
    from .agent_session import VoiceSession
    from .turn import EndpointingOptions, TurnDetectionMode

_OPTION_KEYS = (SessionKwargs.__required_keys__ | SessionKwargs.__optional_keys__) - {
    "conn_options"
}
_NODES = ("llm_node", "tts_node", "stt_node", "transcription_node")
_MODELS = ("stt", "tts", "vad")


def initialize(
    session: VoiceSession[Context_T],
    *,
    stt: NotGivenOr[stt.STT | str] = NOT_GIVEN,
    vad: NotGivenOr[VAD | None] = NOT_GIVEN,
    llm: NotGivenOr[llm.LLM | llm.RealtimeModel | str] = NOT_GIVEN,
    tts: NotGivenOr[tts.TTS | str] = NOT_GIVEN,
    tools: NotGivenOr[list[llm.Tool | llm.Toolset]] = NOT_GIVEN,
    context: Context_T | None = None,
    thread: Thread | None = None,
    video_sampler: NotGivenOr[VoiceActivityVideoSampler | None] = NOT_GIVEN,
    conn_options: NotGivenOr[SessionConnectOptions] = NOT_GIVEN,
    turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
    endpointing_opts: NotGivenOr[EndpointingOptions] = NOT_GIVEN,
    stt_context_options: NotGivenOr[STTContextOptions | None] = NOT_GIVEN,
    **options: Any,
) -> None:
    """A voice session over the Rust core.

    Args:
        stt, vad, llm, tts: The providers, or a LiveKit Inference model string. ``vad``
            defaults to ``inference.VAD(model="silero")``; ``vad=None`` opts out.
        tools: Tools shared by every agent of the session.
        context, thread: Read by tools through ``RunContext``.
        video_sampler: How often camera frames reach the model; ``None`` sends every frame.
        turn_detection, endpointing_opts: The session's turn completion and endpointing
            policy, as ``update_options`` changes them.
        options: The session options the core resolves: ``turn_handling``,
            ``stt_context_options``, ``max_tool_steps``, ``use_tts_aligned_transcript``,
            ``tts_text_transforms``, ``min_consecutive_speech_delay``, ``expressive``,
            ``aec_warmup_duration``, ``ivr_detection``, ``user_away_timeout``,
            ``transcription_timeout``, ``session_close_transcript_timeout``.
    """
    if unknown := options.keys() - _OPTION_KEYS:
        raise TypeError(f"unexpected VoiceSession options: {sorted(unknown)}")
    listening = dict(options.get("turn_handling") or {})
    if is_given(turn_detection):
        listening["turn_detection"] = turn_detection
    if is_given(endpointing_opts):
        listening["endpointing"] = endpointing_opts
    if listening:
        options["turn_handling"] = listening
    session._conn_options = conn_options or SessionConnectOptions()
    session._options = _core.SessionOptions(
        cast(SessionKwargs, {**_given(**options), "conn_options": session._conn_options})
    )
    session._stt_context = stt_context_options if is_given(stt_context_options) else None

    session._stt = inference.STT.from_model_string(stt) if isinstance(stt, str) else stt or None
    session._llm = llm_from_model_string(llm) if isinstance(llm, str) else llm or None
    session._tts = inference.TTS.from_model_string(tts) if isinstance(tts, str) else tts or None
    session._tools = tools if is_given(tools) else []
    turn_handling = options.get("turn_handling") or {}
    detection = turn_handling.get("turn_detection", NOT_GIVEN)
    defaults = _core.default_session_models(
        is_given(vad), is_given(detection), isinstance(session._llm, RealtimeModel)
    )
    session._using_default_vad = defaults.vad_model is not None
    session._vad = (
        inference.VAD(model=defaults.vad_model)
        if defaults.vad_model is not None
        else vad
        if is_given(vad)
        else None
    )
    if not is_given(detection):
        detection = inference.TurnDetector() if defaults.turn_detector else None
    # the gateway detector's stream is the core's own detector; a plugin's is seen through a view
    session._turn_detector = (
        None
        if isinstance(detection, str | None)
        else detection.stream()
        if isinstance(detection, inference.TurnDetector)
        else TurnDetector(detection)
    )
    session._turn_detection_mode = detection if isinstance(detection, str) else None
    session._interruption_mode = (turn_handling.get("interruption") or {}).get("mode")
    session._video_sampler = (
        video_sampler if is_given(video_sampler) else VoiceActivityVideoSampler()
    )

    # `ctx.context` on this call; `None` when the session has no application context
    session._context = cast("Context_T", context)
    session._thread = thread
    # `ctx.execution_id` on this call: one conversational task across its turns
    session._execution_id = shortuuid("execution_")
    session._store = None

    session._owner = _core._core_SessionOwner(session._options)
    session._room = None
    session._amd = None
    session._options.update_providers(
        {"stt": session._stt, "tts": session._tts, "vad": session._vad}
    )
    # an `update_agent` the core has not answered yet: the agent the session is handing to
    session._chat_ctx = ChatContext.empty()
    session._agent_state = "initializing"
    session._user_state = "listening"
    session._input = io.AgentInput(session)
    session._output = io.AgentOutput(session)
    # what a job's session report reads
    session._started_at = None
    session._event_snapshots = deque()
    session._event_retention = _core._core_EventRetention(capture_content_enabled())
    session._recording_options = cast(
        RecordingOptions,
        dataclasses.asdict(
            _core.plan_session_start(job_default=False, another_session_is_primary=False).recording
        ),
    )
    # the core's span tree, materialised here: the open spans by the core's id, and the
    # root's context the job's telemetry parents its own spans under
    session._spans = {}
    session._session_span = None
    session._root_span_context = None
    session._text_input_cb = None
    # the session's audio recording, written by the core; finished once it closes
    session._recorder = None
    # the pipeline callbacks registered on this session, by step name
    session._steps = {}
    session._recording_path = None
    session._audio_input_options = None
    session._tasks = set()
    # the handles the core announced, for their done callbacks (``speech_done``)
    session._speech_handles = weakref.WeakValueDictionary()


class SessionAdapter:
    def __init__(self, session: VoiceSession[Any]) -> None:
        self.session = session

    def _register(self, step: str, f: Callable[..., Any]) -> None:
        session = self.session
        if session._core is not None:
            raise RuntimeError(f"register {step} before the session starts")
        if step in session._steps:
            raise RuntimeError(f"this session already has a {step}")
        session._steps[step] = f

    def _interruption_detector(
        self, agent: Agent, model: Any
    ) -> inference.AdaptiveInterruptionDetector | None:
        """The adaptive interruption detector, when the reference's gate builds one."""
        session = self.session
        # the developer's STT: a fallback chain's core leg reports its capabilities as a dict
        stt_caps = getattr(session._stt, "capabilities", None)
        has_vad = session._vad is not None
        realtime = isinstance(model, RealtimeModel)
        verdict = _core.resolve_interruption_detector(
            stt_aligned_transcript=bool(stt_caps.aligned_transcript) if stt_caps else None,
            stt_streaming=stt_caps.streaming if stt_caps else None,
            realtime=realtime,
            rt_server_turn_detection=realtime
            and _core.rt_server_turn_detection(
                caps_turn_detection=model.capabilities.turn_detection,
                can_disable_turn_detection=model.capabilities.can_disable_turn_detection,
                vad=has_vad,
                turn_detection=session._turn_detection_mode,
                streaming_detector=session._turn_detector is not None,
                interruption_given=bool(session._interruption_mode),
            ),
            has_vad=has_vad,
            turn_detection=session._turn_detection_mode,
            allow_interruptions=session.options["interruption"]["enabled"],
            agent_mode=None,
            session_mode=session._interruption_mode,
            hosted=is_hosted(),
            dev_mode=is_dev_mode(),
        )
        if verdict["message"]:
            log = logger.warning if verdict["level"] == "warning" else logger.info
            log(verdict["message"])
        if not verdict["build"]:
            return None
        try:
            return inference.AdaptiveInterruptionDetector()
        except ValueError as e:
            logger.warning("failed to create AdaptiveInterruptionDetector", extra={"error": str(e)})
            return None

    def _pool(self, agent: Agent) -> list[llm.Tool | llm.Toolset]:
        session = self.session
        return [*agent.tools, *session._tools]

    def _voice_agent(self, agent: Agent, *, task: bool = False) -> _NativeAgent:
        """``agent`` as its callbacks run on this session: its definition over its tools and the
        session's it lacks, its ``AgentCallbacks`` and the context they run in, whose binding is this
        session's."""
        session = self.session
        from .agent_host import _voice_tools

        captured = contextvars.copy_context()
        binding = _VoiceBinding(session, agent, task=task)
        captured.run(_BindingContextVar.set, binding)
        return _core.VoiceAgent(
            agent._definition(_voice_tools(binding)),
            agent._host,
            (asyncio.get_running_loop(), captured),
            toolset_scope=id(agent),
        )

    def _frame(self, agent: Agent, *, task: bool = False) -> dict[str, Any]:
        """``agent`` as the core installs it (``update_agent``'s keywords); the core has its
        toolsets acquired by the native registry before Python reflects their tools. ``task``:
        it runs a ``VoiceTask``."""
        session = self.session
        # its tools, instructions and hooks are what the core runs from now on
        agent._config.freeze()
        pool = self._pool(agent)
        frame: dict[str, Any] = {
            "instructions": agent._config.instructions,
            "dynamic_instructions": (
                _DynamicInstructions(session, agent) if agent._config.dynamic_instructions else None
            ),
            "agent": agent.name,
            "agent_host": agent,
            "tools_host": pool,
            "voice_agent": self._voice_agent(agent, task=task),
        }
        # every leg a node may wrap is given, so the previous frame's node never stays on it; the
        # session's STT and TTS stay as they are, under the agent's nodes
        binding = _VoiceBinding(session, agent)
        for leg in ("stt", "llm", "tts"):
            value = getattr(binding, leg)
            if (leg == "llm" and is_given(agent._llm)) or not isinstance(value, llm.RealtimeModel):
                frame[leg] = _bindings.core_leg(value)
        return {**frame, **self._nodes(agent)}

    def _nodes(self, agent: Agent) -> dict[str, Any]:
        """The registered nodes, as the core opens them for ``agent``, each call in its
        binding."""
        session = self.session
        registered = [n for n in _NODES if n in session._steps]
        if not registered:
            return {}
        context = contextvars.copy_context()
        context.run(_BindingContextVar.set, _VoiceBinding(session, agent))
        # taken in the agent's binding: each node call runs in a copy of it
        return dict.fromkeys(registered, context.run(_core.AgentNodes, _SessionNodes(session)))

    def _legs(self) -> SessionLegs:
        session = self.session
        return {
            "stt": session._stt is not None,
            "vad": session._vad is not None,
            "using_default_vad": session._using_default_vad,
            "llm": isinstance(session._llm, llm.LLM),
            "tts": session._tts is not None,
            "realtime": isinstance(session._llm, llm.RealtimeModel),
            "detector": None,
            "adaptive_detector": False,
            "audio_enabled": True,
            "audio_output": True,
            "sink_can_pause": True,
        }

    def _install_models(self, models: dict[str, Any]) -> None:
        session = self.session
        for leg, model in models.items():
            setattr(session, f"_{leg}", model)
        if "vad" in models:
            session._using_default_vad = False
        # Plugin proxies capture the running host loop. Before startup, retain only SDK objects;
        # the startup adapter binds their effective values when the native owner asks for them.
        if models and session._core is not None:
            nodes = self._nodes(session.current_agent)
            session._owner.install_providers(
                ProviderBindings(
                    **{
                        **{leg: _bindings.core_leg(model) for leg, model in models.items()},
                        **{
                            f"{leg}_node": nodes[f"{leg}_node"]
                            for leg in models
                            if f"{leg}_node" in nodes
                        },
                    }
                )
            )

    @staticmethod
    def _split(options: dict[str, Any]) -> tuple[dict[str, Any], UpdateKwargs]:
        """``options`` as its providers and its settings."""
        models = {k: v for k, v in options.items() if k in _MODELS}
        return models, cast("UpdateKwargs", {k: v for k, v in options.items() if k not in _MODELS})

    def update_agent(self, agent: Agent, *, chat_ctx: ChatContext | None = None) -> None:
        """Hand the conversation to ``agent``; the session and its history stay. ``chat_ctx`` is
        the context ``agent`` works on instead of the conversation carried forward."""
        session = self.session
        frame = None
        if session._core is not None:
            fields = self._frame(agent)
            if chat_ctx is not None:
                fields["chat_ctx"] = chat_ctx.items
            frame = UpdateAgentArgs(**fields)
        session._owner.update_agent(agent, frame)
        if session._core is None and chat_ctx is not None:
            session._chat_ctx = chat_ctx.copy()

    def say(
        self,
        text: str | AsyncIterable[str],
        *,
        audio: NotGivenOr[AsyncIterable[rtc.AudioFrame]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        add_to_chat_ctx: bool = True,
    ) -> SpeechHandle:
        session = self.session
        kwargs = {
            "allow_interruptions": allow_interruptions if is_given(allow_interruptions) else None,
            "add_to_chat_ctx": add_to_chat_ctx,
            "origin": _origin(),
        }
        if is_given(audio):
            assert isinstance(text, str), "say(audio=) takes the words it speaks as a str"
            core = session._running.say_audio(text, audio, **kwargs)
        elif isinstance(text, str):
            core = session._running.say(text, **kwargs)
        else:
            core = session._running.say_stream(text, **kwargs)
        return SpeechHandle(core)

    def generate_reply(
        self,
        *,
        user_input: NotGivenOr[str] = NOT_GIVEN,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[str]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
    ) -> SpeechHandle:
        """Reply now. ``tools`` narrows the offered tools to these names for this reply."""
        session = self.session
        toolset: dict[str, Any] = {}
        if is_given(tools):
            pool = self._pool(session.current_agent)
            specs = tool_specs(pool)
            specs = [specs[index] for index in _core.reply_tool_indices(specs, tools)]
            toolset = {
                "tools": specs,
                "tools_host": pool,
                "voice_agent": self._voice_agent(session.current_agent),
            }
        core = session._running.generate_reply(
            user_input=user_input if is_given(user_input) else None,
            instructions=instructions if is_given(instructions) else None,
            allow_interruptions=allow_interruptions if is_given(allow_interruptions) else None,
            tool_choice=tool_choice if is_given(tool_choice) else None,
            chat_ctx=chat_ctx.items if is_given(chat_ctx) else None,
            on_enter=_origin() == "hook" and _hook_name.get(None) == "on_enter",
            origin=_origin(),
            **toolset,
        )
        return SpeechHandle(core)

    async def _start_ivr_detection(self, transcript: str | None = None) -> None:
        """Start IVR detection: the DTMF tool joins the session's tools and the core's navigator
        runs, counting ``transcript`` (AMD's greeting) as heard."""
        session = self.session
        from ..beta import send_dtmf_events

        if send_dtmf_events not in session._tools:
            session._tools.append(send_dtmf_events)
        session._running.ivr_start(
            tools=tool_specs([send_dtmf_events]), tools_host=[send_dtmf_events]
        )
        if transcript is not None:
            session._running.ivr_final(transcript)

    def configure(self, **options: Any) -> None:
        session = self.session
        models, turn = self._split(_provider_options(options))
        self._install_models(
            session._owner.update_options(
                turn, models, interruption_mode_given="mode" in (turn.get("interruption") or {})
            )
        )

    @contextlib.asynccontextmanager
    async def override(self, **options: Any) -> AsyncIterator[None]:
        session = self.session
        models, turn = self._split(_provider_options(options))
        # One override identity for turn settings and runtime-owned provider references.
        layer = None
        failure: BaseException | None = None
        try:
            override = session._owner.override_options(
                turn, models, interruption_mode_given="mode" in (turn.get("interruption") or {})
            )
            layer = override.layer
            self._install_models(override.providers)
            yield
        except BaseException as error:
            failure = error
            raise
        finally:
            # a release never raises: a closed session has nothing left to restore
            if layer is not None:
                try:
                    self._install_models(session._owner.release_override(layer))
                except _core.SessionClosing as refused:
                    # a closed session has nothing left to restore; the block's own error is the
                    # one to raise
                    if failure is not None:
                        failure.add_note(f"restoring the overridden providers: {refused}")


def _provider_options(options: dict[str, Any]) -> dict[str, Any]:
    options = _given(**options)
    if "stt" in options:
        options["stt"] = _stt_model(options["stt"])
    if "tts" in options:
        options["tts"] = _tts_model(options["tts"])
    return options


def _given(**options: Any) -> dict[str, Any]:
    return {k: v for k, v in options.items() if is_given(v)}


def _stt_model(model: NotGivenOr[stt.STT | str | None]) -> NotGivenOr[stt.STT | None]:
    return inference.STT.from_model_string(model) if isinstance(model, str) else model


def _tts_model(model: NotGivenOr[tts.TTS | str | None]) -> NotGivenOr[tts.TTS | None]:
    return inference.TTS.from_model_string(model) if isinstance(model, str) else model


@function_tool(name="record_keyterms")
async def _record_keyterms(pending: list[str], confirm: list[str], remove: list[str]) -> None:
    """Update the STT keyterms based on the latest transcript.

    Args:
        pending: Distinctive terms seen but not yet trusted — tracked, not applied.
        confirm: Pending terms the transcript has now corroborated — applied.
        remove: Only a spelling the user corrected away; applied terms are otherwise sticky.
    """
    # only used to elicit a structured tool call; never executed


def _detection_llm(stt_context: STTContextOptions | None) -> LLM | None:
    """The detection LLM ``stt_context_options`` asks for: the ``LLM`` given, else the model the
    core names (``_core.resolve_keyterm_detection``) through the inference gateway. ``None`` with
    detection off, or when the model cannot be created (detection is then skipped)."""
    config = (stt_context or {}).get("keyterm_detection")
    resolved = _core.resolve_keyterm_detection(config)
    if not resolved.enabled:
        return None
    if (model := resolved.model) is None:
        llm = config["llm"]  # type: ignore[index]
        return llm if isinstance(llm, LLM) else None
    try:
        from ..inference import LLM as InferenceLLM

        return InferenceLLM.from_model_string(model)
    except Exception:  # noqa: BLE001 — never let detection setup break the session
        logger.warning("keyterm detection: could not create detection LLM %r; skipping", model)
        return None
