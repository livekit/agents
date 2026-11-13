"""The avatar bases: the framework's ``start`` around a plugin's one ``create_session`` attempt,
and the forwarding loop of an avatar fed in process."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import jwt
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, APIStatusError, VoiceSession
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarSession,
    AvatarStart,
    ForwardingAvatarSession,
    QueueAudioOutput,
)

# one retry, its wait under the real-wait bound
RETRY_ONCE = APIConnectOptions(max_retry=1, retry_interval=0.01)
CREDENTIALS = {
    "livekit_url": "wss://lk.test",
    "livekit_api_key": "lk-key",
    "livekit_api_secret": "lk-secret-with-enough-entropy-32b",
}


class _Room(rtc.Room):
    """A room ``room-1`` the agent is connected to as ``agent-1``."""

    @property
    def name(self) -> str:
        return "room-1"

    def isconnected(self) -> bool:
        return True

    @property
    def local_participant(self) -> rtc.LocalParticipant:
        return SimpleNamespace(identity="agent-1")  # type: ignore[return-value]


class _Avatar(AvatarSession):
    def __init__(self, *answers: str | Exception) -> None:
        super().__init__(conn_options=RETRY_ONCE, **CREDENTIALS)
        self.answers = list(answers)
        self.tokens: list[str] = []
        self.ended: list[str] = []

    @property
    def provider(self) -> str:
        return "fake"

    async def create_session(self, ctx: AvatarStart) -> str:
        self.tokens.append(ctx.livekit_token())
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def end_session(self, session_id: str) -> None:
        self.ended.append(session_id)

    def audio_output(self, ctx: AvatarStart) -> QueueAudioOutput:
        return QueueAudioOutput()


# Catches: an avatar start without retry, a token that does not publish for the agent, and a
# provider session left running at close (or ended twice by concurrent closes).
async def test_start_retries_a_provider_failure_and_close_ends_the_session_once() -> None:
    avatar = _Avatar(APIStatusError("busy", status_code=503), "session-1")
    await avatar.start(VoiceSession(vad=None), _Room())
    assert len(avatar.tokens) == 2
    claims = jwt.decode(avatar.tokens[0], options={"verify_signature": False})
    assert claims["sub"] == "fake-avatar-agent"
    assert claims["kind"] == "agent"
    assert claims["attributes"]["lk.publish_on_behalf"] == "agent-1"

    await asyncio.gather(avatar.aclose(), avatar.aclose())
    assert avatar.ended == ["session-1"]


# Catches: an answer the plugin could not parse retried (a second session created) or raised as
# a bare exception instead of an APIError.
async def test_a_failure_outside_the_api_taxonomy_is_final_and_closes_the_avatar() -> None:
    avatar = _Avatar(KeyError("conversation_id"), "never")
    with pytest.raises(APIError) as failed:
        await avatar.start(VoiceSession(vad=None), _Room())
    assert (failed.value.kind, failed.value.retryable) == ("protocol", False)
    assert isinstance(failed.value.__cause__, KeyError)
    assert len(avatar.tokens) == 1
    assert avatar.ended == []
    await avatar.wait_for_join(timeout=0)  # closed: returns at once


class _Forwarding(ForwardingAvatarSession):
    def __init__(self) -> None:
        super().__init__(**CREDENTIALS)
        self.closed = asyncio.Event()

    @property
    def provider(self) -> str:
        return "forwarding"

    async def create_session(self, ctx: AvatarStart) -> str:
        return "session-1"

    async def forward(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        pass

    async def clear(self) -> None:
        raise APIError("the provider refused the interrupt", kind="server")

    async def aclose(self) -> None:
        await super().aclose()
        self.closed.set()


# Catches: a forwarding loop that logs a provider failure and goes on with a dead avatar.
async def test_a_forwarding_failure_closes_the_avatar() -> None:
    avatar = _Forwarding()
    await avatar.start(VoiceSession(vad=None), _Room())
    output = avatar._audio_output
    assert isinstance(output, QueueAudioOutput)
    output.emit("clear_buffer")
    await asyncio.wait_for(avatar.closed.wait(), 2)
    await avatar.wait_for_join(timeout=0)
