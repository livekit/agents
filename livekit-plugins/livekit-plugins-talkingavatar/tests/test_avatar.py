import asyncio
import json
import struct
import time
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web

from livekit import api, rtc
from livekit.agents import AgentSession, APIConnectOptions, utils
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.plugins.talkingavatar import AvatarSession, TalkingAvatarException
from livekit.plugins.talkingavatar.avatar import SAMPLE_RATE, _AvatarAudioOutput

pytestmark = pytest.mark.unit


def _frame(seconds: float) -> rtc.AudioFrame:
    return rtc.AudioFrame.create(SAMPLE_RATE, 1, int(SAMPLE_RATE * seconds))


def _output() -> tuple[_AvatarAudioOutput, utils.aio.Chan, list, list]:
    ch = utils.aio.Chan[bytes | str]()
    out = _AvatarAudioOutput(ch)
    started: list = []
    finished: list = []
    out.on("playback_started", started.append)
    out.on(
        "playback_finished",
        lambda ev: finished.append((round(ev.playback_position, 3), ev.interrupted)),
    )
    return out, ch, started, finished


def _sent(ch: utils.aio.Chan) -> list:
    msgs = []
    while True:
        try:
            msgs.append(ch.recv_nowait())
        except utils.aio.channel.ChanEmpty:
            return msgs


async def test_segment_finishes_once_heard():
    out, ch, started, finished = _output()
    for _ in range(3):
        await out.capture_frame(_frame(0.1))
    out.flush()

    msgs = _sent(ch)
    assert len(msgs) == 3
    play_at, rate = struct.unpack_from("<dI", msgs[0])
    assert (play_at, rate, len(msgs[0])) == (0, SAMPLE_RATE, 12 + 2 * 2400)

    assert finished == []
    out._on_event({"type": "speech_started"})
    assert len(started) == 1
    out._on_event({"type": "speech_ended", "playedMs": 300})
    assert finished == [(0.3, False)]
    await asyncio.wait_for(out.wait_for_playout(), 1)


async def test_queue_running_dry_mid_segment_is_not_the_end():
    out, _, started, finished = _output()
    await out.capture_frame(_frame(0.2))
    out._on_event({"type": "speech_started"})
    out._on_event({"type": "speech_ended", "playedMs": 200})  # the TTS fell behind
    assert finished == []

    await out.capture_frame(_frame(0.2))
    out._on_event({"type": "speech_started"})
    out.flush()
    assert finished == []
    out._on_event({"type": "speech_ended", "playedMs": 200})
    assert (len(started), finished) == (1, [(0.4, False)])


async def test_heard_before_flush_finishes_at_flush():
    out, _, _, finished = _output()
    await out.capture_frame(_frame(0.2))
    out._on_event({"type": "speech_started"})
    out._on_event({"type": "speech_ended", "playedMs": 200})
    out.flush()
    assert finished == [(0.2, False)]


async def test_next_segment_waits_for_its_own_start():
    out, _, started, finished = _output()
    await out.capture_frame(_frame(0.2))
    out.flush()
    out._on_event({"type": "speech_started"})
    await out.capture_frame(_frame(0.2))  # sent while the avatar's run ends: a run of its own
    out.flush()
    out._on_event({"type": "speech_ended", "playedMs": 200})
    assert (len(started), finished) == (1, [(0.2, False)])

    out._on_event({"type": "speech_started"})
    assert len(started) == 2
    out._on_event({"type": "speech_ended", "playedMs": 200})
    assert finished == [(0.2, False), (0.2, False)]


async def test_back_to_back_segments_share_one_run():
    out, _, started, finished = _output()
    for _ in range(2):
        await out.capture_frame(_frame(0.3))
        out.flush()
    out._on_event({"type": "speech_started"})
    out._on_event({"type": "speech_ended", "playedMs": 600})  # one run for both
    assert finished == [(0.3, False), (0.3, False)]
    assert len(started) == 2
    assert abs(started[1].created_at - (time.time() - 0.3)) < 0.05  # it began 0.3 s before the end


async def test_interruption_reports_what_was_heard():
    out, ch, _, finished = _output()
    await out.capture_frame(_frame(1.0))
    out.flush()
    out._on_event({"type": "speech_started"})
    _sent(ch)

    out.clear_buffer()
    out.clear_buffer()  # the agent session clears twice on an interruption
    assert _sent(ch) == [json.dumps({"type": "clear"})]
    assert finished == []

    out._on_event({"type": "cleared", "playedMs": 300})
    assert finished == [(0.3, True)]
    await asyncio.wait_for(out.wait_for_playout(), 1)

    out.clear_buffer()  # nothing playing: nothing to send
    assert _sent(ch) == []


async def test_lost_connection_does_not_hang_the_agent():
    out, ch, _, finished = _output()
    await out.capture_frame(_frame(0.5))
    out.flush()
    out._on_closed()
    assert finished == [(0.0, True)]

    _sent(ch)
    await out.capture_frame(_frame(0.5))
    out.flush()
    assert finished == [(0.0, True), (0.0, True)]
    assert _sent(ch) == []


class _FakeAvatarServer:
    """The Talking Avatar realtime API: "ready" for a good key, close 4001 for a bad one."""

    def __init__(self) -> None:
        self.start_msg: dict | None = None
        self.audio_bytes = 0
        self.close_code: int | None = None

    async def handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.start_msg = json.loads((await ws.receive()).data)
        if self.start_msg["apiKey"] != "tav_good":
            await ws.close(code=4001)
            return ws
        await ws.send_json({"type": "ready", "width": 832, "height": 468, "audio": True})
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                self.audio_bytes += len(msg.data) - 12
            elif json.loads(msg.data)["type"] == "clear":
                await ws.send_json({"type": "cleared", "playedMs": 0})
        self.close_code = ws.close_code
        return ws


def _fake_room() -> SimpleNamespace:
    local = SimpleNamespace(identity="agent-1", kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT)
    return SimpleNamespace(
        name="room-1",
        local_participant=local,
        remote_participants={},
        isconnected=lambda: True,
        on=lambda *a: None,
        off=lambda *a: None,
    )


async def test_session_starts_in_the_room_and_closes():
    server = _FakeAvatarServer()
    app = web.Application()
    app.router.add_get("/", server.handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"ws://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/"  # type: ignore[union-attr]
    lk = {
        "livekit_url": "wss://lk.test",
        "livekit_api_key": "key",
        "livekit_api_secret": "secret" * 6,
    }

    try:
        bad = AvatarSession(
            api_key="tav_bad", api_url=url, conn_options=APIConnectOptions(max_retry=0)
        )
        with pytest.raises(TalkingAvatarException, match="invalid or missing API key"):
            await bad.start(AgentSession(), _fake_room(), **lk)
        await bad.aclose()

        session = AgentSession()
        avatar = AvatarSession(api_key="tav_good", avatar_id="ava7", api_url=url)
        await avatar.start(session, _fake_room(), **lk)

        start = server.start_msg
        assert start is not None
        assert (start["avatar"], start["livekitUrl"], start["sampleRate"]) == (
            "ava7",
            "wss://lk.test",
            24000,
        )
        assert start["audio"] == {"publish": True, "trackName": "avatar-audio"}
        claims = api.TokenVerifier("key", "secret" * 6).verify(start["token"])
        assert claims.identity == avatar.avatar_identity
        assert claims.kind == "agent"
        assert claims.attributes == {ATTRIBUTE_PUBLISH_ON_BEHALF: "agent-1"}
        assert (
            claims.video.room == "room-1"
            and claims.video.room_join
            and not claims.video.can_subscribe
        )

        assert session.output.audio is avatar._audio_output
        await session.output.audio.capture_frame(_frame(0.1))
        session.output.audio.flush()
        await asyncio.sleep(0.2)
        assert server.audio_bytes == 2 * 2400

        await avatar.aclose()
        await asyncio.sleep(0.1)
        assert server.close_code == 1000  # billing stops
    finally:
        await runner.cleanup()
