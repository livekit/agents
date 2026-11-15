from collections.abc import Callable
from typing import Literal, Never, assert_type

from livekit.agents import llm, stt, tts
from livekit.agents.voice import VoiceSession
from livekit.agents.voice.avatar import AvatarSession


def default_events(
    speech: stt.STT,
    voice: tts.TTS,
    model: llm.LLM,
    avatar: AvatarSession,
    duplex: llm.DuplexSession,
) -> None:
    assert_type(speech, stt.STT[Never])
    assert_type(voice, tts.TTS[Never])
    assert_type(model, llm.LLM[Never])
    assert_type(avatar, AvatarSession[Never])
    assert_type(duplex, llm.DuplexSession[Never])


def callback_signatures(session: VoiceSession[None]) -> None:
    @session.on("user_state_changed")
    def state(event: object, /) -> None:
        pass

    assert_type(state, Callable[[object], None])


def extended_events(session: llm.RealtimeSession[Literal["provider_event"]]) -> llm.RealtimeSession:
    return session


class RelayingChain(stt.FallbackAdapter[Literal["member_event"]]):
    """A chain that relays its members' own event: fails while ``stt.FallbackAdapter`` emits
    only its availability event."""

    def relay(self, payload: object) -> None:
        self.emit("member_event", payload)
        self.emit("stt_availability_changed", payload)
        self.emit("unknown_event", payload)  # type: ignore[arg-type]


def chain_events(chain: stt.FallbackAdapter) -> None:
    assert_type(chain, stt.FallbackAdapter[Never])
