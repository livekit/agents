from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class AMDCategory(str, Enum):
    HUMAN = "human"
    MACHINE_IVR = "machine-ivr"
    MACHINE_SCREENING = "machine-screening"
    MACHINE_VM = "machine-vm"
    MACHINE_UNAVAILABLE = "machine-unavailable"
    UNCERTAIN = "uncertain"


class AMDReason(str, Enum):
    """Why a prediction was published or why the run completed."""

    PREDICTION = "prediction"
    LATE_PREDICTION = "late_prediction"
    REUSED = "reused"
    SUPERSEDED = "superseded"
    INFERENCE_TIMEOUT = "inference_timeout"
    INFERENCE_ERROR = "inference_error"
    INTERNAL_ERROR = "internal_error"
    FINISHED = "finished"
    MAX_UNCERTAIN_TURNS = "max_uncertain_turns"
    TIMEOUT = "timeout"
    IDLE_TIMEOUT = "idle_timeout"
    CANCELLED = "cancelled"
    PARTICIPANT_MISSING = "participant_missing"
    PARTICIPANT_DISCONNECTED = "participant_disconnected"
    AGENT_CHANGED = "agent_changed"


class AMDPredictionEvent(BaseModel):
    type: Literal["amd_prediction"] = "amd_prediction"
    speech_duration: float
    category: AMDCategory
    reason: AMDReason
    transcript: str
    delay: float
    turn_id: int = 0
    prev_turn_category: AMDCategory | None = None
    prev_stage_category: AMDCategory | None = None
    inference_duration: float | None = None
    voicemail_message_played: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def state_changed(self) -> bool:
        return self.category != (self.prev_turn_category or AMDCategory.UNCERTAIN)

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


class IvrMenuOption(BaseModel):
    label: str
    dtmf: str = ""
    spoken_response: str = ""


class AMDMenuObservedEvent(BaseModel):
    """Best-effort observation. This event never authorizes an action."""

    type: Literal["amd_menu_observed"] = "amd_menu_observed"
    session_id: str
    turn_id: int
    menu: str
    options: list[IvrMenuOption] = Field(default_factory=list)
    extraction_duration: float
