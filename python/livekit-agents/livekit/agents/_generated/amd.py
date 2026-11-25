# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/voice/amd.rs

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from inspect import get_annotations
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from livekit.agents.llm import LLM
    from livekit.agents.stt import STT
    from livekit.agents.voice.amd import _DecisionModel

    from .connect import APIConnectOptions
    from .tool_options import ToolSpec


# Record declarations:


@unique
class AMDCategory(StrEnum):
    """A per-turn prediction or a stage. `uncertain` and `wait` keep the stage."""

    HUMAN = "human"
    MACHINE_IVR = "machine-ivr"
    MACHINE_SCREENING = "machine-screening"
    MACHINE_VM = "machine-vm"
    MACHINE_UNAVAILABLE = "machine-unavailable"
    WAIT = "wait"
    """The stage stays, and the reply is skipped."""
    UNCERTAIN = "uncertain"
    """The stage stays, and a reply is allowed."""


@unique
class AMDReason(StrEnum):
    """Why a prediction was published or why the run completed."""

    PREDICTION = "prediction"
    """A model prediction was released normally."""
    REUSED = "reused"
    """An empty turn reused the latest prediction without a new request."""
    INFERENCE_TIMEOUT = "inference_timeout"
    """The turn's inference deadline passed, so the prediction is uncertain in the current
    stage. Repeated inference timeouts can also complete the run.
    """
    INFERENCE_ERROR = "inference_error"
    """Classification failed or returned unusable output; the prediction is uncertain in the
    current stage.
    """
    INTERNAL_ERROR = "internal_error"
    """An internal task or prediction handler failed, ending the run."""
    FINISHED = "finished"
    """AMD reached a human or machine-unavailable category."""
    MAX_UNCERTAIN_TURNS = "max_uncertain_turns"
    """The limit of consecutive uncertain predictions was reached."""
    TIMEOUT = "timeout"
    """The time limit since listening started was reached."""
    IDLE_TIMEOUT = "idle_timeout"
    """The call stayed silent and inactive for the current stage's idle interval."""
    CANCELLED = "cancelled"
    """The caller closed AMD or its session before detection completed."""
    PARTICIPANT_MISSING = "participant_missing"
    """The target participant or audio track was unavailable during listening setup."""
    PARTICIPANT_DISCONNECTED = "participant_disconnected"
    """The target participant disconnected before detection completed."""
    AGENT_CHANGED = "agent_changed"
    """The session switched agents during AMD."""


@unique
class AMDLifecycle(StrEnum):
    """Where one AMD run is: `initialized` until entered, `pending` until it listens, `active`
    while it listens, `finished` once it completed.
    """

    INITIALIZED = "initialized"
    PENDING = "pending"
    ACTIVE = "active"
    FINISHED = "finished"


class AMDPredictionEvent(BaseModel):
    """`amd_prediction`: at most one per committed turn."""

    type: Literal["amd_prediction"] = "amd_prediction"
    speech_duration: float
    """Participant speech in this turn, in seconds."""
    category: AMDCategory
    """Prediction for this turn. ``uncertain`` and ``wait`` keep the current stage."""
    reason: AMDReason
    transcript: str
    """The session transcript, else AMD's transcript."""
    delay: float
    """Time from commit to this event, in seconds."""
    turn_id: int = 0
    """AMD's turn id: local to the run, from 1. Ids can skip a turn whose request a newer turn
    cancelled.
    """
    stage: AMDCategory = AMDCategory("uncertain")
    """Stage after this prediction. Never ``wait``."""
    inference_duration: float | None = None
    """Model time for ``prediction`` events, else ``None``."""

    @property
    def is_human(self) -> bool:
        """The prediction is ``human``."""
        from .. import _core

        return _core._core_amd_prediction_is_human(self)

    @property
    def is_machine(self) -> bool:
        """The prediction is one of the four machine categories."""
        from .. import _core

        return _core._core_amd_prediction_is_machine(self)


class AMDCompletedEvent(BaseModel):
    """`amd_completed`: once per run, after cleanup."""

    type: Literal["amd_completed"] = "amd_completed"
    category: AMDCategory
    """The call's category. Once the voicemail message has played, a goodbye that
    ends the box does not make it ``machine-unavailable``: the category is the stage
    before it, ``machine-vm``, or ``machine-ivr`` after a post-message menu.
    """
    reason: AMDReason
    voicemail_message_played: bool = False
    """The voicemail message played to its end while AMD was active."""
    last_prediction: AMDPredictionEvent | None = None
    """The last classified turn's prediction; None when the run ended before one."""

    @property
    def is_error(self) -> bool:
        """Whether AMD stopped without detecting the call, for example after an internal error
        or a missing participant. Timeouts, idle completions, and a participant
        disconnect are normal outcomes; read ``category`` for those.
        """
        from .. import _core

        return _core._core_amd_completed_is_error(self)


class IVRMenuOption(BaseModel):
    """One option of an observed IVR menu."""

    label: str
    dtmf: str = ""
    spoken_response: str = ""


class AMDMenuObservedEvent(BaseModel):
    """Best-effort observation. This event never authorizes an action."""

    type: Literal["amd_menu_observed"] = "amd_menu_observed"
    session_id: str
    turn_id: int
    menu: str
    options: list[IVRMenuOption] = Field(default_factory=list)
    extraction_duration: float
    """Model time of the extraction, in seconds."""


AmdEvent: TypeAlias = AMDPredictionEvent | AMDMenuObservedEvent | AMDCompletedEvent
"""What a run tells its host, in order: each prediction and menu, then the completion once."""


@dataclass
class AMDOptions:
    """`AMD(...)`'s behavior options, with their defaults. Model and participant selection are the
    constructor's own arguments.
    """

    wait_until_answered: bool = True
    """Discard pre-answer audio from AMD and the session. When False, listen to subscribed SIP
    early media.
    """
    screening_instructions: str | None = "Call state: automated call screening. Answer the screening assistant's latest prompt briefly. Use what you know about who you are and why you are calling. Then wait for its next prompt."
    """Instructions for each screening turn. None completes AMD when the call enters
    screening, without a reply.
    """
    voicemail_instructions: str | None = "Call state: voicemail. Deliver one concise, self-contained message. State who you are, why you are calling, and the next step for the recipient."
    """Instructions for one successfully played voicemail message. None completes AMD at the
    first voicemail prediction, without a message.
    """
    ivr_instructions: str | None = "Call state: an automated phone menu. Use the participant's actual prompt and the purpose of this call to choose the next step. Use send_dtmf_events for an explicit keypad choice, or give a short spoken answer when requested. Do not invent a menu option. After the action, wait for the next prompt."
    """Instructions for each IVR turn. The SDK supplies a DTMF tool only to IVR reply
    generations. None completes AMD when the call enters IVR, without a reply or menu
    extraction.
    """
    human_instructions: str | None = "Call state: a human has answered. The latest participant turn is from the human, not the automated system. Earlier automated prompts no longer apply. Respond to the human's latest message and continue the call normally. Do not resume a response to an earlier automated prompt."
    """Temporary instructions for the first human turn after a machine stage. Skipped if no
    machine stage preceded the human. None keeps the Agent's own instructions.
    """
    opening_text: str | None = None
    """Fixed text that the agent says first, when AMD starts listening. Set at most one of
    opening_text and opening_instructions.
    """
    opening_instructions: str | None = None
    """Instructions for a reply that the agent generates and says first, when AMD starts
    listening. The reply cannot call tools.
    """
    idle_timeout: float = 10.0
    """Silent, inactive time before AMD completes outside voicemail, in seconds."""
    voicemail_idle_timeout: float = 10.0
    """Silent, inactive time in voicemail, in seconds."""
    timeout: float = 60.0
    """Hard limit from the start of listening, in seconds."""
    inference_timeout: float = 2.0
    """Prediction deadline per committed turn, in seconds. Late results are ignored."""
    machine_silence_threshold: float = 1.5
    """Continuous participant silence before a machine reply plays, in seconds. Zero disables
    the extra silence wait.
    """
    max_uncertain_turns: int = 3
    """Consecutive uncertain predictions before completion with the current stage."""
    max_inference_timeouts: int = 3
    """Prediction timeouts before completion. A valid prediction resets the count."""


class AMDOptionsKwargs(TypedDict, total=False):
    """`AMDOptions`'s fields as keyword arguments."""

    wait_until_answered: bool
    """Discard pre-answer audio from AMD and the session. When False, listen to subscribed SIP
    early media.
    """
    screening_instructions: str | None
    """Instructions for each screening turn. None completes AMD when the call enters
    screening, without a reply.
    """
    voicemail_instructions: str | None
    """Instructions for one successfully played voicemail message. None completes AMD at the
    first voicemail prediction, without a message.
    """
    ivr_instructions: str | None
    """Instructions for each IVR turn. The SDK supplies a DTMF tool only to IVR reply
    generations. None completes AMD when the call enters IVR, without a reply or menu
    extraction.
    """
    human_instructions: str | None
    """Temporary instructions for the first human turn after a machine stage. Skipped if no
    machine stage preceded the human. None keeps the Agent's own instructions.
    """
    opening_text: str | None
    """Fixed text that the agent says first, when AMD starts listening. Set at most one of
    opening_text and opening_instructions.
    """
    opening_instructions: str | None
    """Instructions for a reply that the agent generates and says first, when AMD starts
    listening. The reply cannot call tools.
    """
    idle_timeout: float
    """Silent, inactive time before AMD completes outside voicemail, in seconds."""
    voicemail_idle_timeout: float
    """Silent, inactive time in voicemail, in seconds."""
    timeout: float
    """Hard limit from the start of listening, in seconds."""
    inference_timeout: float
    """Prediction deadline per committed turn, in seconds. Late results are ignored."""
    machine_silence_threshold: float
    """Continuous participant silence before a machine reply plays, in seconds. Zero disables
    the extra silence wait.
    """
    max_uncertain_turns: int
    """Consecutive uncertain predictions before completion with the current stage."""
    max_inference_timeouts: int
    """Prediction timeouts before completion. A valid prediction resets the count."""


@dataclass(kw_only=True, slots=True)
class AmdLlm:
    """One classifier or STT model a run uses, as the host resolved it."""

    model: LLM
    inference: bool
    """The model reaches only LiveKit Inference: its requests carry AMD's client feature header."""


@dataclass(kw_only=True, slots=True)
class AmdDecision:
    """A decision model AMD classifies with, as the host resolved it."""

    model: _DecisionModel
    inference: bool
    """The model reaches only LiveKit Inference: its requests carry AMD's client feature header."""


@dataclass(kw_only=True, slots=True)
class AmdStt:
    """AMD's own streaming STT, as the host resolved it."""

    model: STT
    inference: bool
    """The model reaches only LiveKit Inference: its connections carry AMD's client feature
    header.
    """


AmdClassifier: TypeAlias = AmdLlm | AmdDecision
"""What AMD classifies with: an LLM, which also reads menus, or a decision model."""


@dataclass(kw_only=True, slots=True)
class AmdEntry:
    """What `AMD.__aenter__` hands the session: everything one run needs beyond the session.
    The host closes the models it created once the run completed.
    """

    options: AMDOptions
    classifier: AmdClassifier
    stt: AmdStt | None = None
    participant: str | None = None
    """The participant to listen to; `None` listens to the one the session's room I/O links."""
    dtmf_tool: ToolSpec
    """AMD's built-in `send_dtmf_events`, which the host runs."""
    llm_conn_options: APIConnectOptions
    """The session's `llm_conn_options`: classifier and menu requests use them."""
    stt_conn_options: APIConnectOptions
    """The session's `stt_conn_options`: AMD's STT stream uses them."""


@dataclass(kw_only=True, slots=True)
class AmdChoiceRequest:
    """One `evaluate` call with a single `Choice` decision."""

    input: str
    """`evaluate(input=...)`: the request state and the transcript."""
    name: str
    """The decision's name."""
    prompt: str
    """The choice's prompt."""
    options: list[str]
    """The choice's options, in order. They carry no descriptions: a short definition next to a
    choice outweighed the prompt's conditions.
    """
    conn_options: APIConnectOptions
    extra_headers: dict[str, str]
    """`extra_headers`."""


AMDCategory.__module__ = "livekit.agents.voice.amd"
AMDReason.__module__ = "livekit.agents.voice.amd"
AMDLifecycle.__module__ = "livekit.agents.voice.amd"
AMDPredictionEvent.__module__ = "livekit.agents.voice.amd"
AMDCompletedEvent.__module__ = "livekit.agents.voice.amd"
IVRMenuOption.__module__ = "livekit.agents.voice.amd"
AMDMenuObservedEvent.__module__ = "livekit.agents.voice.amd"
AMDOptions.__annotations__ = get_annotations(AMDOptions, eval_str=True)
AMDOptions.__module__ = "livekit.agents.voice.amd"
AMDOptionsKwargs.__module__ = "livekit.agents.voice.amd"
AmdLlm.__module__ = "livekit.agents.voice.amd"
AmdDecision.__module__ = "livekit.agents.voice.amd"
AmdStt.__module__ = "livekit.agents.voice.amd"
AmdEntry.__module__ = "livekit.agents.voice.amd"
AmdChoiceRequest.__module__ = "livekit.agents.voice.amd"


DEFAULT_AMD_LLM_MODEL: Final = "google/gemma-4-31b-it"
"""The classifier AMD selects when `llm` is not given and LiveKit Cloud credentials exist."""
DEFAULT_AMD_STT_MODEL: Final = "cartesia/ink-whisper"
"""AMD's own STT when `stt` is not given and LiveKit Cloud credentials exist."""
