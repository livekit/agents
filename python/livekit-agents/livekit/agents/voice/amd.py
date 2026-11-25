"""Multi-turn answering-machine detection (AMD) on a running ``VoiceSession``.

The core runs the detection: the session's reply gate, the stage machine, the classifier, AMD's
own STT and the callee's answer. This module selects the models, hands the run to the session,
closes the models it created, runs the DTMF tool and delivers the run's events to listeners.
"""

from __future__ import annotations

import asyncio
import inspect
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import Unpack

from livekit import rtc

from .. import _core, inference, llm, stt
from .._generated import public_voice_amd as _public
from .._generated.amd import DEFAULT_AMD_LLM_MODEL, DEFAULT_AMD_STT_MODEL
from .._generated.public_voice_amd import *  # noqa: F403
from .._generated.public_voice_amd import (
    AmdChoiceRequest,
    AMDCompletedEvent,
    AmdDecision,
    AmdEntry,
    AMDLifecycle,
    AmdLlm,
    AMDOptions,
    AMDOptionsKwargs,
    AMDPredictionEvent,
    AmdStt,
)
from ..log import logger
from ..stt import STT
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from ..utils import aio, is_given

if TYPE_CHECKING:
    from .._core import AmdRun
    from ..llm.tool_context import Tool
    from .agent_session import VoiceSession

__all__ = ["AMD"]
__all__ += _public.__all__

AMDEventTypes = Literal["amd_prediction", "amd_completed", "amd_menu_observed"]


async def _choose(model: llm.DecisionModel, request: AmdChoiceRequest) -> str | None:
    """One ``evaluate`` call with one ``Choice``: the chosen option, or ``None`` when the
    response holds no choice for it. A failure propagates: an ``APIError``, ``ValueError`` or
    ``TimeoutError`` is an ``inference_error`` prediction."""
    kwargs: dict[str, Any] = {"conn_options": request.conn_options}
    if request.extra_headers:
        kwargs["extra_headers"] = dict(request.extra_headers)
    response = await model.evaluate(
        input=request.input,
        decisions={
            request.name: llm.Choice(
                prompt=request.prompt, options=dict.fromkeys(request.options, "")
            )
        },
        **kwargs,
    )
    result = response.results.get(request.name)
    return result.value if isinstance(result, llm.ChoiceResult) else None


def _decision_label(model: llm.DecisionModel) -> str:
    return f"{model.provider}/{model.model}"


def _reaches_inference(model: object) -> bool:
    """Whether every request ``model`` makes reaches LiveKit Inference (a fallback chain only
    when each member does): those carry AMD's client feature header."""
    if isinstance(model, llm.FallbackAdapter):
        return all(_reaches_inference(member) for member in model._members)
    return isinstance(model, (inference.LLM, inference.STT))


def _label(model: Any) -> str:
    return f"{getattr(model, 'provider', 'unknown')}/{getattr(model, 'model', 'unknown')}"


def _amd_tools() -> list[Tool]:
    from ..beta import send_dtmf_events

    return [send_dtmf_events]


class AMD(rtc.EventEmitter[AMDEventTypes]):
    """Multi-turn answering-machine detection.

    Start a VoiceSession before entering AMD. Enter AMD before creating a SIP
    participant. execute() waits for completion, not the first machine prediction,
    unless that stage's instructions are None.
    The SDK owns classification and stage control. Models can use any provider.

    AMD uses the session's current Agent. Realtime models require client-side
    turn detection, session or AMD STT, and per-response tool
    control. Agent handoffs during AMD are not supported. Normal hooks, interruptions,
    playback, and StopResponse stay in the session. Menu events are informational
    and never execute actions.
    AMD enables interruptions while it runs, then restores the settings when detection
    finishes. Direct speech generation during AMD is not supported. To let the agent speak
    first, set opening_text or opening_instructions.

    Example:
        async with AMD(session) as amd:
            result = await amd.execute()

    Args:
        session: Started session whose participant audio and client-side EOT to use.
        llm: Classification and menu model. When omitted, use
            ``google/gemma-4-31b-it`` if LiveKit Cloud credentials are
            available; otherwise use the current Agent's LLM. Pass None to
            always use the current Agent's LLM, for example to keep AMD within
            the quota of the session's models. A string selects a LiveKit
            Inference model. AMD logs the selected models when it starts.
            A decision model that supports choice decisions classifies, but AMD
            then skips menu extraction. Calls use
            ``session.conn_options.llm_conn_options``. Supplied models stay
            open when AMD completes.
        stt: Optional streaming STT model for AMD. When omitted, use
            ``cartesia/ink-whisper`` if LiveKit Cloud credentials are available;
            otherwise use only the session transcript. Pass None to always use
            only the session transcript, with no extra STT usage. With session
            STT configured, the classifier sees both transcripts of each turn. A string
            selects a LiveKit Inference model.
        participant_identity: Select the participant before placing an outbound call.
            When omitted, AMD listens to the participant that room I/O links.
        options: ``AMDOptions``' fields: the stage instructions, the opening, the timeouts,
            ``machine_silence_threshold``, ``max_uncertain_turns``,
            ``max_inference_timeouts`` and ``wait_until_answered``.
    """

    _DEFAULT_LLM_MODEL: str = DEFAULT_AMD_LLM_MODEL
    _DEFAULT_STT_MODEL: str = DEFAULT_AMD_STT_MODEL

    def __init__(
        self,
        session: VoiceSession[Any],
        *,
        llm: NotGivenOr[llm.LLM | llm.DecisionModel | str | None] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | str | None] = NOT_GIVEN,
        participant_identity: NotGivenOr[str] = NOT_GIVEN,
        **options: Unpack[AMDOptionsKwargs],
    ) -> None:
        super().__init__()
        self._options = AMDOptions(**options)
        _core._core_amd_check_options(self._options)
        llm_source = "custom" if llm is not None else "session"
        stt_source = "custom" if stt is not None else "session"
        if not is_given(llm) or not is_given(stt):
            auto_select = _core._core_amd_auto_models()
            if not is_given(llm):
                llm = self._DEFAULT_LLM_MODEL if auto_select else None
                llm_source = "default" if auto_select else "session"
            if not is_given(stt):
                stt = self._DEFAULT_STT_MODEL if auto_select else None
                stt_source = "default" if auto_select else "session"

        self._session = session
        self._model_sources = (llm_source, stt_source)
        self._llm_model = llm if isinstance(llm, str) else None
        self._llm: Any = None if isinstance(llm, str) else llm
        self._stt_model = stt if isinstance(stt, str) else None
        self._stt: STT | None = None if isinstance(stt, str) else stt
        if self._stt is not None and not self._stt.capabilities.streaming:
            raise ValueError("amd requires a streaming STT")
        self._participant_identity = participant_identity
        # the models AMD created from strings, closed once the run completed
        self._owned: list[Any] = []
        self._lifecycle = AMDLifecycle.INITIALIZED
        self._run: AmdRun | None = None
        self._completion: asyncio.Future[AMDCompletedEvent] | None = None
        self._tasks = aio.TaskSet()
        self._pump: asyncio.Task[None] | None = None

    @property
    def lifecycle(self) -> AMDLifecycle:
        """Current lifecycle state of this AMD run."""
        if self._run is None or self._lifecycle is AMDLifecycle.FINISHED:
            return self._lifecycle
        return self._run.lifecycle

    async def __aenter__(self) -> AMD:
        if self._lifecycle is not AMDLifecycle.INITIALIZED:
            raise RuntimeError("use a new AMD instance for each run")
        session = self._session
        native = session._running
        agent = session.current_agent
        agent_llm = agent.llm if is_given(agent.llm) else session.llm
        classifier = self._llm or agent_llm
        if self._llm_model is None and not isinstance(classifier, (llm.LLM, llm.DecisionModel)):
            raise ValueError("amd requires an LLM or a decision model for classification")
        if (
            isinstance(classifier, llm.DecisionModel)
            and "choice" not in classifier.capabilities.decision_kinds
        ):
            raise ValueError("amd requires a decision model that supports choice decisions")
        if (
            self._options.opening_text is not None
            and session.tts is None
            and not (
                isinstance(agent_llm, llm.RealtimeModel) and agent_llm.capabilities.supports_say
            )
        ):
            raise ValueError(
                "amd opening_text requires a TTS model or a realtime model that supports say()"
            )
        if self._options.opening_instructions is not None and agent_llm is None:
            raise ValueError("amd opening_instructions requires an LLM on the session")

        self._lifecycle = AMDLifecycle.PENDING
        try:
            if self._llm_model is not None:
                self._llm = inference.LLM.from_model_string(self._llm_model)
                self._owned.append(self._llm)
            if self._stt_model is not None:
                self._stt = inference.STT.from_model_string(self._stt_model)
                self._owned.append(self._stt)
            if self._stt is not None and not self._stt.capabilities.streaming:
                raise ValueError("amd requires a streaming STT")
            classifier = self._llm or agent_llm
            entry = self._entry(classifier)
            self._run = await native.enter_amd(entry)
        except BaseException:
            self._lifecycle = AMDLifecycle.FINISHED
            await self._close_owned()
            raise

        llm_source, stt_source = self._model_sources
        # AMD defaults use their own model quotas, apart from the session's models
        source = next((s for s in self._model_sources if s != "default"), "default")
        logger.info(
            f"amd initialized with {source} models",
            extra={
                "llm": _label(classifier),
                "llm_source": llm_source,
                "stt": _label(self._stt) if self._stt is not None else "session transcript only",
                "stt_source": stt_source,
            },
        )
        self._completion = asyncio.get_running_loop().create_future()
        session._amd = self
        session._amd_tools = _amd_tools()
        # the pump ends with the run's completion, which the session's close also brings
        self._pump = self._tasks.spawn(self._deliver(self._run), name="amd_events")
        return self

    def _entry(self, classifier: Any) -> AmdEntry:
        from .._bindings import tool_specs

        conn = self._session.conn_options
        if isinstance(classifier, llm.LLM):
            model: AmdLlm | AmdDecision = AmdLlm(
                model=classifier, inference=_reaches_inference(classifier)
            )
        else:
            model = AmdDecision(model=classifier, inference=False)
        return AmdEntry(
            options=self._options,
            classifier=model,
            stt=(
                AmdStt(model=self._stt, inference=_reaches_inference(self._stt))
                if self._stt is not None
                else None
            ),
            participant=(
                self._participant_identity if is_given(self._participant_identity) else None
            ),
            dtmf_tool=tool_specs(_amd_tools())[0],
            llm_conn_options=conn.llm_conn_options or DEFAULT_API_CONNECT_OPTIONS,
            stt_conn_options=conn.stt_conn_options or DEFAULT_API_CONNECT_OPTIONS,
        )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Complete AMD with ``cancelled`` if it is still running, and wait for its cleanup."""
        if self._run is None or self._completion is None:
            return
        await self._run.aclose()
        await asyncio.shield(self._completion)

    async def execute(self) -> AMDCompletedEvent:
        """Wait for completion and cleanup; every call returns the same ``AMDCompletedEvent``."""
        if self._completion is None:
            raise RuntimeError("enter AMD before use")
        return await asyncio.shield(self._completion)

    def emit(self, event: AMDEventTypes, *args: Any) -> None:
        """Call each listener of ``event``. Unlike ``rtc.EventEmitter``, a listener's error
        propagates: a failing prediction or menu listener ends the run with
        ``internal_error``."""
        for callback in list(self._events.get(event, ())):
            params = inspect.signature(callback).parameters.values()
            if any(p.kind is p.VAR_POSITIONAL for p in params):
                callback(*args)
                continue
            positional = [
                p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            callback(*args[: len(positional)])

    async def _deliver(self, run: AmdRun) -> None:
        """Hands each event to the listeners, in order, until the completion."""
        try:
            while (event := await run.next_event()) is not None:
                if isinstance(event, AMDCompletedEvent):
                    await self._complete(event)
                    return
                name: AMDEventTypes = (
                    "amd_prediction"
                    if isinstance(event, AMDPredictionEvent)
                    else "amd_menu_observed"
                )
                try:
                    self.emit(name, event)
                except Exception as error:
                    logger.error(
                        f"amd {name.removeprefix('amd_')} handler failed",
                        extra={"error_type": type(error).__name__},
                    )
                    run.listener_failed()
                if isinstance(event, AMDPredictionEvent):
                    logger.info(
                        "amd prediction",
                        extra={
                            "turn_id": event.turn_id,
                            "category": event.category.value,
                            "reason": event.reason.value,
                        },
                    )
            # the run's events ended without the completion (its queue overflowed)
            await self._complete(await run.execute())
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if self._completion is not None and not self._completion.done():
                self._completion.set_exception(error)
            raise

    async def _complete(self, result: AMDCompletedEvent) -> None:
        self._lifecycle = AMDLifecycle.FINISHED
        await self._close_owned()
        if self._session._amd is self:
            self._session._amd = None
        if self._completion is not None and not self._completion.done():
            self._completion.set_result(result)
        try:
            self.emit("amd_completed", result)
        except Exception:
            # the run is over; a failing listener must not fail cleanup or session close
            logger.exception("amd_completed listener failed")

    async def _close_owned(self) -> None:
        owned, self._owned = self._owned, []
        for error in await asyncio.gather(
            *(model.aclose() for model in owned), return_exceptions=True
        ):
            if isinstance(error, BaseException):
                logger.warning(
                    "amd resource cleanup failed", extra={"error_type": type(error).__name__}
                )
