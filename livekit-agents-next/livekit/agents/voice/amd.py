"""``AMD``: the answering-machine detector a call attaches to its session.

The core runs the detector (``_core.VoiceSession.attach_amd``): the wait on the room for the
callee's answer, the classifier, its timers, the authorization hold, the IVR navigator's close,
AMD's own STT and the interrupt on a machine verdict. This class resolves the models and hands a
``machine-ivr`` verdict to the navigator. The classifier's tools here are the objects a host LLM is
handed beside the core's specs, and are never executed.
"""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from .. import _bindings, _core, inference
from .._generated.amd import AMDCategory, AMDPredictionEvent, DetectionOptions
from ..inference._utils import default_api_key, default_api_secret
from ..job import server_setting
from ..llm import LLM as _LLM
from ..llm.tool_context import function_tool
from ..log import logger
from ..stt import STT as _STT
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import EventEmitter, is_given

if TYPE_CHECKING:
    from ..inference import LLMModels
    from .agent_session import VoiceSession

__all__ = [
    "AMD",
    "AMDCategory",
    "AMDPredictionEvent",
    "AMD_PROMPT",
    "DetectionOptions",
]

AMD_PROMPT: str = _core.amd_constants()["prompt"]


@function_tool
async def save_prediction(label: AMDCategory) -> None:
    """Save the prediction to the verdict."""


@function_tool
async def postpone_termination(seconds: float) -> str:
    """Postpone the termination of the classification task.
    Use when the transcript is ambiguous and more audio is expected.

    Args:
        seconds: Additional seconds to wait (max 10).
    """
    return ""


CLASSIFIER_TOOLS = [save_prediction, postpone_termination]


_CONSTANTS: dict[str, Any] = _core.amd_constants()


EVALUATED_LLM_MODELS: set[str] = set(_CONSTANTS["evaluated_llm_models"])


EVALUATED_STT_MODELS: set[str] = set(_CONSTANTS["evaluated_stt_models"])


# `DetectionOptions`' keys as the core's config names them
_CONFIG_KEYS = {
    "human_speech_threshold": "human_speech_threshold",
    "human_silence_threshold": "human_silence_threshold",
    "machine_silence_threshold": "machine_silence_threshold",
    "no_speech_threshold": "no_speech_threshold",
    "timeout": "detection_timeout",
    "prompt": "prompt",
}


class AMD(EventEmitter[Literal["amd_prediction"]]):
    """Answering Machine Detection (AMD).

    Detects whether an outbound call is answered by a human or a machine, classifying the
    greeting as ``human``, ``machine-ivr``, ``machine-vm``, ``machine-unavailable`` or
    ``uncertain``.

    Start AMD before creating the SIP participant so no audio is missed::

        async with AMD(session, llm="openai/gpt-4.1-mini") as detector:
            await ctx.api.sip.create_sip_participant(...)  # wait_until_answered=True
            result = await detector.execute()

    Args:
        session: The :class:`VoiceSession` to wire AMD to.
        llm: LLM used for greeting classification, an :class:`LLM` or an inference model
            string. When omitted, the gateway's default model with LiveKit Cloud credentials,
            else the session's LLM. ``None`` always reuses the session's LLM.
        stt: STT of AMD's own, an :class:`STT` or an inference model string. When omitted, the
            gateway's default model with LiveKit Cloud credentials, else the session's
            transcripts. ``None`` always reuses the session's transcripts.
        interrupt_on_machine: Interrupt the agent's speech when a machine is detected.
        ivr_detection: Start IVR navigation on a ``machine-ivr`` verdict.
        participant_identity: Listen only to this participant's audio track.
        suppress_compatibility_warning: Do not warn about models AMD was not evaluated with.
        detection_options: Overrides of the timing thresholds and the classification prompt.
        wait_until_finished: Once speech was heard, wait for the greeting to end rather than
            emit at ``timeout``.
    """

    def __init__(
        self,
        session: VoiceSession,
        *,
        llm: NotGivenOr[_LLM | LLMModels | str | None] = NOT_GIVEN,
        stt: NotGivenOr[_STT | str | None] = NOT_GIVEN,
        interrupt_on_machine: bool = True,
        ivr_detection: bool = True,
        participant_identity: NotGivenOr[str] = NOT_GIVEN,
        suppress_compatibility_warning: bool = False,
        detection_options: NotGivenOr[DetectionOptions] = NOT_GIVEN,
        wait_until_finished: bool = True,
    ) -> None:
        super().__init__()
        if not is_given(llm) or not is_given(stt):
            auto_select = _core.amd_auto_select_models(
                server_setting("LIVEKIT_URL") or "",
                default_api_key(None),
                default_api_secret(None),
            )
            if not is_given(llm):
                llm = _CONSTANTS["default_llm_model"] if auto_select else None
            if not is_given(stt):
                stt = _CONSTANTS["default_stt_model"] if auto_select else None

        self._llm_config: _LLM | str | None = llm
        self._session = session
        self._interrupt_on_machine = interrupt_on_machine
        self._ivr_detection = ivr_detection
        self._wait_until_finished = wait_until_finished
        self._suppress_compatibility_warning = suppress_compatibility_warning
        self._participant_identity = participant_identity or None
        self._stt: _STT | None = inference.STT(stt) if isinstance(stt, str) else stt
        self._detection_options: DetectionOptions = (
            detection_options if is_given(detection_options) else {}
        )

        self._attached = False
        self._result: AMDPredictionEvent | None = None
        self._verdict_ready = asyncio.Event()

        if not suppress_compatibility_warning and self._stt is not None:
            _warn_if_not_evaluated(self._stt.model, "stt")

    @property
    def enabled(self) -> bool:
        return self._attached

    @property
    def pending(self) -> bool:
        return self._attached and self._result is None

    @property
    def started(self) -> bool:
        return self._attached and self._session._running.amd_listening()

    async def execute(self) -> AMDPredictionEvent:
        """Wait for the verdict and return it.

        Speech playout stays unauthorized until the verdict; the core then interrupts the agent
        on a machine (``interrupt_on_machine``) and resumes. A ``machine-ivr`` verdict starts
        IVR navigation first (``ivr_detection``).
        """
        if self._attached:
            await self._verdict_ready.wait()

        if not self._result:
            raise RuntimeError("amd closed before a result was available")

        result = self._result
        if result.category == AMDCategory.MACHINE_IVR and self._ivr_detection:
            await self._session._start_ivr_detection(transcript=result.transcript)
        self._session._running.amd_execute()
        return result

    async def __aenter__(self) -> AMD:
        await self._run(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._attached:
            self._session._running.amd_close()
        self._attached = False
        self._verdict_ready.set()
        if self._session._amd is self:
            self._session._amd = None

    async def _run(self, session: VoiceSession) -> None:
        if self._attached:
            logger.warning("AMD already running, skipping")
            return

        llm = self._resolve_llm(session)
        if llm is None:
            raise ValueError(
                "AMD classifier could not be resolved, please provide a compatible model"
            )
        if session.options["ivr_detection"]:
            logger.warning("session level ivr_detection will be disabled when AMD is used")

        provided = self._detection_options
        config: dict[str, Any] = {
            key: provided[name]  # type: ignore[literal-required]
            for name, key in _CONFIG_KEYS.items()
            if name in provided
        }
        config["max_endpointing_delay"] = _core.amd_resolve_max_endpointing_delay(
            provided.get("max_endpointing_delay"), session.options["endpointing"]["max_delay"]
        )
        config["source"] = "amd_stt" if self._stt is not None else "stt"
        config["wait_until_finished"] = self._wait_until_finished
        config["interrupt_on_machine"] = self._interrupt_on_machine
        config["ivr_detection"] = self._ivr_detection
        session._running.attach_amd(
            config,
            _bindings.core_leg(llm),
            tools=CLASSIFIER_TOOLS,
            stt=_bindings.core_leg(self._stt) if self._stt is not None else None,
            participant_identity=self._participant_identity,
        )
        self._attached = True
        self._result = None
        session._amd = self

    def _on_prediction(self, result: AMDPredictionEvent) -> None:
        """The core's ``amd_result``."""
        self._result = result
        logger.info(
            "amd prediction",
            extra={
                "category": result.category.value,
                "reason": result.reason,
                "speech_duration": result.speech_duration,
                "delay": result.delay,
                "lk.pii.transcript": result.transcript,
            },
        )
        self._verdict_ready.set()
        self.emit("amd_prediction", result)

    def _resolve_llm(self, session: VoiceSession) -> _LLM | None:
        llm: _LLM | None = None
        if isinstance(self._llm_config, str):
            llm = inference.LLM(self._llm_config)
        elif isinstance(self._llm_config, _LLM):
            llm = self._llm_config
        elif isinstance(session.llm, _LLM):
            llm = session.llm
        if not self._suppress_compatibility_warning and llm is not None:
            _warn_if_not_evaluated(llm.model, "llm")
        return llm


def _warn_if_not_evaluated(model: str | None, kind: str) -> None:
    if model and not _core.amd_is_evaluated_model(model, kind):
        logger.warning(
            "%s model %s hasn't been evaluated with our benchmark, it might not be compatible "
            "with amd. Set `suppress_compatibility_warning=True` to silence this warning.",
            kind,
            model,
        )
