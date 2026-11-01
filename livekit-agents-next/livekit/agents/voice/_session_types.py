"""Public voice-session configuration and callback signatures."""

from __future__ import annotations

from collections.abc import AsyncIterable, Callable, Coroutine
from typing import Any, Literal

from typing_extensions import TypeVar

from livekit import rtc
from livekit.agents._generated.recording_options import (
    RecordingOptions as RecordingOptions,
)

from .. import llm, stt
from .._generated.connect import SessionConnectOptions as SessionConnectOptions
from .._generated.session_wire import (
    DEFAULT_HOOK_BOUND,
    DEFAULT_SILENT_FPS,
    DEFAULT_SPEAKING_FPS,
)
from ..llm import (
    ChatContext,
)
from ..types import (
    FlushSentinel,
    TranscriptChunk,
)
from .agent import (
    ModelSettings,
)
from .events import (
    RunContext,
    UserTurnExceededEvent,
)

TextTransforms = Literal["filter_markdown", "filter_emoji"]
"""A transform the core applies to the text it sends to the TTS."""


# on_exit and the other callbacks are awaited by the core this long before it moves on;
# on_user_turn_completed is awaited until it returns
_HOOK_BOUND = DEFAULT_HOOK_BOUND

# the session's pipeline callbacks (voice-session.md "Customizing the pipeline"): each receives
# the call's `RunContext`, whose `agent` is the agent holding the call
_C = TypeVar("_C")
_Item = TypeVar("_Item")
_Node = AsyncIterable[_Item] | Coroutine[Any, Any, AsyncIterable[_Item] | None]
LLMNode = Callable[
    [RunContext[_C], ChatContext, list[llm.Tool], ModelSettings],
    AsyncIterable[llm.ChatChunk | str | FlushSentinel]
    | Coroutine[Any, Any, AsyncIterable[llm.ChatChunk | str | FlushSentinel] | str | None],
]
STTNode = Callable[
    [RunContext[_C], AsyncIterable[rtc.AudioFrame], ModelSettings], _Node[stt.SpeechEvent]
]
TTSNode = Callable[[RunContext[_C], AsyncIterable[str], ModelSettings], _Node[rtc.AudioFrame]]
TranscriptionNode = Callable[
    [RunContext[_C], AsyncIterable[TranscriptChunk], ModelSettings], _Node[str | TranscriptChunk]
]
UserTurnCompletedHook = Callable[
    [RunContext[_C], ChatContext, llm.ChatMessage], Coroutine[Any, Any, None]
]
UserTurnExceededHook = Callable[[RunContext[_C], UserTurnExceededEvent], Coroutine[Any, Any, None]]


class VoiceActivityVideoSampler:
    """Sample the camera at ``speaking_fps`` while the user speaks, ``silent_fps`` otherwise."""

    def __init__(
        self, *, speaking_fps: float = DEFAULT_SPEAKING_FPS, silent_fps: float = DEFAULT_SILENT_FPS
    ) -> None:
        self.speaking_fps = speaking_fps
        self.silent_fps = silent_fps


_CallbackT = TypeVar("_CallbackT", bound=Callable[..., object])


for _type in (RecordingOptions, VoiceActivityVideoSampler):
    _type.__module__ = "livekit.agents.voice.agent_session"
