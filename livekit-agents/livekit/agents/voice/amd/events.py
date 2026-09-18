from enum import Enum, auto
from typing import Literal

from pydantic import BaseModel, Field


class AMDLifecycle(Enum):
    INITIALIZED = auto()
    PENDING = auto()
    ACTIVE = auto()
    FINISHED = auto()


class AMDCategory(str, Enum):
    HUMAN = "human"
    MACHINE_IVR = "machine-ivr"
    MACHINE_SCREENING = "machine-screening"
    MACHINE_VM = "machine-vm"
    MACHINE_UNAVAILABLE = "machine-unavailable"
    WAIT = "wait"  # state remains, reply skipped
    UNCERTAIN = "uncertain"  # state remains, reply allowed


class AMDReason(str, Enum):
    """Why a prediction was published or why the run completed."""

    PREDICTION = "prediction"
    """A model prediction was released normally."""
    REUSED = "reused"
    """An empty turn reused the latest prediction without a new request."""
    INFERENCE_TIMEOUT = "inference_timeout"
    """The turn's inference deadline passed, so the prediction uses the current stage.

    Repeated inference timeouts can also complete the run.
    """
    INFERENCE_ERROR = "inference_error"
    """Classification failed or returned unusable output; the prediction uses the current stage."""
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


class AMDPredictionEvent(BaseModel):
    type: Literal["amd_prediction"] = "amd_prediction"
    speech_duration: float
    category: AMDCategory
    """Prediction for this turn. ``uncertain`` and ``wait`` keep the current stage."""
    reason: AMDReason
    transcript: str
    delay: float
    turn_id: int = 0
    stage: AMDCategory = AMDCategory.UNCERTAIN
    """Stage after this prediction. Never ``wait``."""
    state_changed: bool = False
    """Whether this prediction started a new stage."""
    prev_turn_category: AMDCategory | None = None
    prev_stage_category: AMDCategory | None = None
    inference_duration: float | None = None
    voicemail_message_played: bool = False

    @property
    def is_human(self) -> bool:
        return self.category == AMDCategory.HUMAN

    @property
    def is_machine(self) -> bool:
        return self.category in (
            AMDCategory.MACHINE_SCREENING,
            AMDCategory.MACHINE_IVR,
            AMDCategory.MACHINE_VM,
            AMDCategory.MACHINE_UNAVAILABLE,
        )


class AMDCompletedEvent(BaseModel):
    type: Literal["amd_completed"] = "amd_completed"
    category: AMDCategory
    reason: AMDReason
    turn_id: int
    transcript: str
    prev_turn_category: AMDCategory | None = None
    prev_stage_category: AMDCategory | None = None
    voicemail_message_played: bool = False


class IVRMenuOption(BaseModel):
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
