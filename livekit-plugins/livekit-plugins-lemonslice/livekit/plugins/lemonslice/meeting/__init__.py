from __future__ import annotations

from dataclasses import dataclass

from livekit.agents import NotGivenOr, utils
from livekit.agents.voice import room_io

from .audio import MeetingAudioInput, stream_meeting_audio, stream_meeting_relay
from .chat import MeetingChatRelay, format_chat_user_input
from .codec import MeetingChatMessage, deserialize_chat

__all__ = [
    "JoinMeetingResult",
    "MeetingAudioInput",
    "MeetingChatMessage",
    "MeetingChatRelay",
    "deserialize_chat",
    "format_chat_user_input",
    "meeting_room_options",
    "stream_meeting_audio",
    "stream_meeting_relay",
]


@dataclass(frozen=True)
class JoinMeetingResult:
    """Result of adding an active LemonSlice session to an external meeting."""

    agent_audio_websocket_url: str
    meeting_bot_id: str


def meeting_room_options(
    *,
    audio_output: NotGivenOr[bool] = False,
    **kwargs: object,
) -> room_io.RoomOptions:
    """RoomOptions for ``AgentSession.start`` after :meth:`AvatarSession.join_meeting`.

    Disables LiveKit room audio I/O — meeting audio is fed via :class:`MeetingAudioInput`.
    """
    opts: dict[str, object] = {
        "audio_input": False,
        **kwargs,
    }
    if utils.is_given(audio_output):
        opts["audio_output"] = audio_output
    return room_io.RoomOptions(**opts)
