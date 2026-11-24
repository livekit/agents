"""The avatar bases: the framework's ``start`` around a plugin's one ``create_session`` attempt,
and the forwarding loop of an avatar fed in process."""

from __future__ import annotations

import asyncio

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
from tests.fakes import AVATAR_CREDENTIALS, AgentRoom, until

# one retry, its wait under the real-wait bound
RETRY_ONCE = APIConnectOptions(max_retry=1, retry_interval=0.01)


class _Avatar(AvatarSession):
    def __init__(self, *answers: str | Exception) -> None:
        super().__init__(conn_options=RETRY_ONCE, **AVATAR_CREDENTIALS)
        self.answers = list(answers)
        self.tokens: list[str] = []
        self.ended: list[str] = []
        self.answering = asyncio.Event()
        self.answering.set()

    @property
    def provider(self) -> str:
        return "fake"

    async def create_session(self, ctx: AvatarStart) -> str:
        self.tokens.append(await ctx.livekit_token())
        await self.answering.wait()
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
    await avatar.start(VoiceSession(vad=None), AgentRoom())
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
        await avatar.start(VoiceSession(vad=None), AgentRoom())
    assert (failed.value.kind, failed.value.retryable) == ("protocol", False)
    assert isinstance(failed.value.__cause__, KeyError)
    assert len(avatar.tokens) == 1
    assert avatar.ended == []
    await avatar.wait_for_join(timeout=0)  # closed: returns at once


# Catches: a start cancelled while the provider creates its session, which left that billable
# session running since its id never reached the close.
async def test_a_start_cancelled_while_creating_ends_the_session_it_created() -> None:
    avatar = _Avatar("session-1")
    avatar.answering.clear()
    start = asyncio.ensure_future(avatar.start(VoiceSession(vad=None), AgentRoom()))
    await until(lambda: avatar.tokens, "the provider is asked for a session")

    start.cancel()
    asyncio.get_running_loop().call_soon(avatar.answering.set)
    with pytest.raises(asyncio.CancelledError):
        await start
    await avatar.aclose()

    assert avatar.ended == ["session-1"]


# Catches: an avatar started again after its close, whose second session no close ended.
async def test_each_start_of_a_reused_avatar_ends_its_own_session() -> None:
    avatar = _Avatar("session-1", "session-2")
    session = VoiceSession(vad=None)
    for _ in range(2):
        await avatar.start(session, AgentRoom())
        await avatar.aclose()

    assert avatar.ended == ["session-1", "session-2"]


class _Forwarding(ForwardingAvatarSession):
    def __init__(self) -> None:
        super().__init__(**AVATAR_CREDENTIALS)
        self.closed = asyncio.Event()
        self.hung_up = asyncio.Event()

    @property
    def provider(self) -> str:
        return "forwarding"

    async def create_session(self, ctx: AvatarStart) -> str:
        return "session-1"

    async def forward(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        pass

    async def clear(self) -> None:
        raise APIError("the provider refused the interrupt", kind="server")

    async def receive(self) -> None:
        await self.hung_up.wait()
        raise APIError("the provider hung up", kind="connection")

    async def aclose(self) -> None:
        await super().aclose()
        self.closed.set()


# Catches: a forwarding loop that logs a provider failure and goes on with a dead avatar.
async def test_a_forwarding_failure_closes_the_avatar() -> None:
    avatar = _Forwarding()
    await avatar.start(VoiceSession(vad=None), AgentRoom())
    output = avatar._audio_output
    assert isinstance(output, QueueAudioOutput)
    output.emit("clear_buffer")
    await asyncio.wait_for(avatar.closed.wait(), 2)
    await avatar.wait_for_join(timeout=0)


# Catches: a provider socket that drops while the agent is silent, leaving a dead avatar open
# until its next speech.
async def test_a_receive_failure_closes_the_avatar() -> None:
    avatar = _Forwarding()
    await avatar.start(VoiceSession(vad=None), AgentRoom())
    avatar.hung_up.set()
    await asyncio.wait_for(avatar.closed.wait(), 2)


class _TokenModeJob:
    """A fake token mode job. It records the claims of each participant token it is asked
    for."""

    def __init__(self) -> None:
        self.asked: list[dict[str, object]] = []

    async def create_participant_token(self, **claims: object) -> str:
        self.asked.append(claims)
        return f"server-token-{len(self.asked)}"


def _start(job: _TokenModeJob, *, key: str = "", secret: str = "") -> AvatarStart:
    from types import SimpleNamespace

    return AvatarStart(
        room=SimpleNamespace(name="call"),  # type: ignore[arg-type]
        agent_session=None,  # type: ignore[arg-type]
        agent_identity="agent-1",
        livekit_url="wss://x",
        conn_options=RETRY_ONCE,
        avatar_identity="fake-avatar",
        avatar_name="Avatar",
        _api_key=key,
        _api_secret=secret,
        _job=job,  # type: ignore[arg-type]
    )


# Starts an avatar in a token mode job, and again with a plugin key and secret. Catches an avatar
# token that is signed with no secret instead of created by the server, or a token that loses its
# claims (publish on behalf of the agent, kind, room). Also catches a plugin key and secret that
# do not sign the token locally.
async def test_avatar_tokens_come_from_the_job_in_token_mode_and_a_key_signs_locally() -> None:
    job = _TokenModeJob()
    start = _start(job)
    assert await start.livekit_token() == "server-token-1"
    assert await start.participant_token(identity="listener") == "server-token-2"
    avatar, listener = job.asked
    assert (avatar["identity"], avatar["name"], avatar["kind"]) == (
        "fake-avatar",
        "Avatar",
        "agent",
    )
    assert avatar["attributes"] == {"lk.publish_on_behalf": "agent-1"}
    grants = avatar["grants"]
    assert (grants.room_join, grants.room) == (True, "call")  # type: ignore[attr-defined]
    assert (listener["identity"], listener["kind"]) == ("listener", None)

    keyed = _start(job, key="plugin-key", secret="plugin-secret-plugin-secret-plugin")
    token = await keyed.livekit_token()
    claims = jwt.decode(token, "plugin-secret-plugin-secret-plugin", algorithms=["HS256"])
    assert (claims["iss"], claims["sub"], claims["kind"]) == ("plugin-key", "fake-avatar", "agent")
    assert len(job.asked) == 2, "a key and secret sign locally"
