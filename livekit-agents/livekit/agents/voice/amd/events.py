from typing import Literal

from pydantic import BaseModel, Field

from .classifier import AMDCategory


class AMDCompletedEvent(BaseModel):
    type: Literal["amd_completed"] = "amd_completed"
    category: AMDCategory
    reason: str
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
