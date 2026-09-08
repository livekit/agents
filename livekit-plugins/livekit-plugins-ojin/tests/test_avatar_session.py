from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fake_stv import FakeSTVClient, make_audio_frame, make_video_frame
from ojin.stv import STVConfig

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.avatar import AudioSegmentEnd, QueueAudioOutput, _types as _avatar_types
from livekit.plugins.ojin import avatar as avatar_mod
from livekit.plugins.ojin.avatar import AvatarSession, _build_avatar_options, _FrameSink
from livekit.plugins.ojin.errors import OjinException

# Hermetic: driven by a fake Ojin client, no network and no credentials.
pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OJIN_API_KEY", raising=False)
    monkeypatch.delenv("OJIN_CONFIG_ID", raising=False)
    monkeypatch.delenv("OJIN_WS_URL", raising=False)


def session(**kwargs: object) -> AvatarSession:
    kwargs.setdefault("api_key", "k")
    kwargs.setdefault("config_id", "c")
    return AvatarSession(**kwargs)  # type: ignore[arg-type]


# --- options derivation -----------------------------------------------------


def test_options_from_a_square_first_frame() -> None:
    options = _build_avatar_options(
        make_video_frame(1024, 1024), audio_sample_rate=24000, video_fps=25
    )

    assert (options.video_width, options.video_height) == (1024, 1024)
    assert options.audio_sample_rate == 24000
    assert options.audio_channels == 1
    assert options.video_fps == 25


def test_options_from_a_non_square_first_frame() -> None:
    options = _build_avatar_options(
        make_video_frame(736, 1216), audio_sample_rate=48000, video_fps=30
    )

    assert (options.video_width, options.video_height) == (736, 1216)
    assert options.audio_sample_rate == 48000
    assert options.video_fps == 30


# --- credentials ------------------------------------------------------------


def test_missing_api_key_raises() -> None:
    with pytest.raises(OjinException, match="OJIN_API_KEY"):
        AvatarSession(config_id="c")


def test_missing_config_id_raises() -> None:
    with pytest.raises(OjinException, match="OJIN_CONFIG_ID"):
        AvatarSession(api_key="k")


def test_env_is_used_when_args_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OJIN_API_KEY", "env-key")
    monkeypatch.setenv("OJIN_CONFIG_ID", "env-config")

    s = AvatarSession()

    assert (s._api_key, s._config_id) == ("env-key", "env-config")


def test_args_beat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OJIN_API_KEY", "env-key")
    monkeypatch.setenv("OJIN_CONFIG_ID", "env-config")

    s = session(api_key="arg-key")

    assert s._api_key == "arg-key"


def test_provider_name() -> None:
    assert session().provider == "ojin"


# --- fade / fps coupling ----------------------------------------------------


def test_sink_fade_follows_stv_config() -> None:
    s = session(stv_config=STVConfig(interrupt_audio_fade_s=0.2, fps=30))

    sink = s._build_sink()

    assert sink._fade_s == 0.2
    assert s._effective_fps() == 30


def test_defaults_without_stv_config() -> None:
    s = session()

    assert s._build_sink()._fade_s == 0.75
    assert s._effective_fps() == 25


# --- fatal error handling ---------------------------------------------------


async def test_server_fatal_shape_reaches_the_handler() -> None:
    """Production's fatal path carries code=; a handler that cannot take it is broken."""
    s = session()
    client = FakeSTVClient()
    s._wire_listeners(client, _FrameSink())

    await client.emit_server_fatal(message="backend down", code="BACKEND_UNAVAILABLE")

    assert client.handler_errors == [], "a listener rejected the SDK's kwargs"
    assert s._fatal_error == "backend down"


async def test_connect_error_shape_reaches_the_handler() -> None:
    s = session()
    client = FakeSTVClient()
    s._wire_listeners(client, _FrameSink())

    await client.emit_connect_error("connect failed")

    assert client.handler_errors == []
    assert s._fatal_error == "connect failed"


async def test_session_ready_handler_accepts_session_data() -> None:
    """A bare Event.set would TypeError on session_data= and be swallowed."""
    s = session()
    client = FakeSTVClient()
    s._wire_listeners(client, _FrameSink())

    await client.emit_session_ready()

    assert client.handler_errors == []
    assert s._ready.is_set()


async def test_fatal_is_first_wins_across_the_error_closed_pair() -> None:
    """The SDK emits ERROR(fatal) then CLOSED back to back on every server error."""
    s = session()
    client = FakeSTVClient()
    s._wire_listeners(client, _FrameSink())

    await client.emit_server_fatal(message="first")
    await client.emit_closed()

    assert s._fatal_error == "first"
    assert s._degrade_task is None, "there is no runner yet, so nothing to degrade"


async def test_plugin_initiated_close_is_ignored() -> None:
    s = session()
    client = FakeSTVClient()
    s._wire_listeners(client, _FrameSink())
    s._closed = True

    await client.emit_closed()

    assert s._fatal_error is None


async def test_non_fatal_error_is_only_logged() -> None:
    s = session()
    client = FakeSTVClient()
    s._wire_listeners(client, _FrameSink())

    await client.emit(avatar_mod.STVEvent.ERROR, message="hiccup", fatal=False)

    assert s._fatal_error is None


# --- lifecycle --------------------------------------------------------------


async def test_aclose_is_idempotent() -> None:
    s = session()
    client = FakeSTVClient()
    s._client = client

    await s.aclose()
    await s.aclose()

    assert client.close_calls == 1


async def test_degrade_survives_its_own_aclose() -> None:
    """_degrade awaits aclose(); if aclose cancelled it, the drain would never start."""
    s = session()
    s._client = FakeSTVClient()
    s._audio_output = QueueAudioOutput(sample_rate=24000, wait_playback_start=True)

    s._degrade_task = asyncio.create_task(s._degrade())
    await asyncio.wait_for(s._degrade_task, 2)

    assert s._degrade_task.cancelled() is False
    assert s._closed is True, "teardown did not run"
    assert s._null_drain_task is not None
    assert not s._null_drain_task.done(), "the code after aclose() never ran"
    await s.aclose()


# --- null drain -------------------------------------------------------------


async def _degraded_session() -> tuple[AvatarSession, QueueAudioOutput]:
    s = session()
    s._client = FakeSTVClient()
    audio_output = QueueAudioOutput(sample_rate=24000, wait_playback_start=True)
    s._audio_output = audio_output
    await s._degrade()
    return s, audio_output


def _frame(ms: int = 100) -> rtc.AudioFrame:
    samples = int(24000 * ms / 1000)
    return rtc.AudioFrame(
        data=bytes(samples * 2), sample_rate=24000, num_channels=1, samples_per_channel=samples
    )


async def test_null_drain_reports_playout_after_teardown() -> None:
    s, audio_output = await _degraded_session()

    await audio_output.capture_frame(_frame(100))
    audio_output.flush()

    ev = await asyncio.wait_for(audio_output.wait_for_playout(), 2)
    assert ev.interrupted is False
    assert ev.playback_position == pytest.approx(0.1, abs=0.01)
    await s.aclose()


async def test_null_drain_reports_started() -> None:
    s, audio_output = await _degraded_session()
    started = asyncio.Event()
    audio_output.on("playback_started", lambda *_: started.set())

    await audio_output.capture_frame(_frame())
    audio_output.flush()
    await asyncio.wait_for(audio_output.wait_for_playout(), 2)

    assert started.is_set()
    await s.aclose()


async def test_interrupt_after_teardown_resolves() -> None:
    """The runner's stale handler cannot be relied on: its gate froze at teardown."""
    s, audio_output = await _degraded_session()

    await audio_output.capture_frame(_frame(100))
    await asyncio.sleep(0.05)
    audio_output.clear_buffer()

    ev = await asyncio.wait_for(audio_output.wait_for_playout(), 2)
    assert ev.interrupted is True
    await s.aclose()


async def test_null_drain_resets_position_between_segments() -> None:
    s, audio_output = await _degraded_session()
    await audio_output.capture_frame(_frame(100))
    audio_output.flush()
    await asyncio.wait_for(audio_output.wait_for_playout(), 2)

    await audio_output.capture_frame(_frame(40))
    audio_output.flush()

    ev = await asyncio.wait_for(audio_output.wait_for_playout(), 2)
    assert ev.playback_position == pytest.approx(0.04, abs=0.01)
    await s.aclose()


# --- watchdog ---------------------------------------------------------------


async def test_watchdog_fires_when_server_frames_stop() -> None:
    """Synthesized sink writes continue forever on a dead transport."""
    s = session(watchdog_timeout=0.05)
    client = FakeSTVClient()
    sink = _FrameSink()
    s._client = client
    s._sink = sink
    s._start_watchdog()

    for _ in range(6):
        await sink.write_video(make_video_frame(fresh=False))  # held frames keep coming
        await asyncio.sleep(0.02)

    assert s._fatal_error is not None
    await s.aclose()


async def test_watchdog_silent_while_server_frames_arrive() -> None:
    s = session(watchdog_timeout=0.1)
    sink = _FrameSink()
    s._client = FakeSTVClient()
    s._sink = sink
    s._start_watchdog()

    for _ in range(5):
        await sink.write_video(make_video_frame(fresh=True))
        await asyncio.sleep(0.02)

    assert s._fatal_error is None
    await s.aclose()


async def test_render_deadline_unwedges_then_second_goes_fatal() -> None:
    s = session(watchdog_timeout=10.0)
    sink = _FrameSink(turn_render_timeout=0.01)
    s._client = FakeSTVClient()
    s._sink = sink

    for _ in range(2):
        sink.note_input_segment_open()
        sink.note_input_audio()
        sink.note_input_segment_end()
        await asyncio.sleep(0.02)
        s._check_render_liveness()

    items = [sink._deque.popleft() for _ in range(len(sink._deque))]
    assert sum(isinstance(i, AudioSegmentEnd) for i in items) == 2
    assert s._fatal_error is not None


async def test_start_rolls_back_when_the_client_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raise anywhere in start() must undo the base session's wiring too."""

    class ExplodingClient(FakeSTVClient):
        async def start(self) -> None:
            raise RuntimeError("connect blew up")

    monkeypatch.setattr(avatar_mod, "OjinSTVClient", ExplodingClient)
    s = session()
    agent_session = SimpleNamespace(
        _started=False,
        output=SimpleNamespace(audio=None),
        on=lambda *a: None,
        off=lambda *a: None,
        emit=lambda *a: None,
    )
    room = SimpleNamespace(isconnected=lambda: False, on=lambda *a: None, off=lambda *a: None)

    with pytest.raises(RuntimeError, match="connect blew up"):
        await s.start(agent_session, room)  # type: ignore[arg-type]

    assert s._closed is True
    assert s._agent_session is None, "base session wiring leaked"


async def test_second_aclose_stops_the_null_drain() -> None:
    """The job's shutdown callback runs aclose again; the drain must not outlive it."""
    s, _ = await _degraded_session()
    drain = s._null_drain_task
    assert drain is not None and not drain.done()

    await s.aclose()

    assert drain.done()
    assert s._null_drain_task is None


async def test_render_stall_streak_resets_when_a_turn_renders() -> None:
    """One stall now and another an hour later must not tear down a healthy session."""
    s = session(watchdog_timeout=10.0)
    sink = _FrameSink(turn_render_timeout=0.01)
    s._client = FakeSTVClient()
    s._sink = sink

    async def stall() -> None:
        sink.note_input_segment_open()
        sink.note_input_audio()
        sink.note_input_segment_end()
        await asyncio.sleep(0.02)
        s._check_render_liveness()

    await stall()
    assert s._fatal_error is None

    # a healthy turn in between
    sink.note_input_segment_open()
    await sink.write_audio(make_audio_frame())
    sink.on_bot_stopped_speaking()
    sink.note_input_segment_end()

    await stall()

    assert s._fatal_error is None, "non-consecutive stalls must not be fatal"


async def test_degrade_does_not_leak_a_drain_when_aclose_races() -> None:
    """A shutdown aclose() during degrade's teardown must not orphan the drain.

    The second call runs its cleanup while the drain is still unstarted; if
    degrade then starts one, nothing is left to stop it and a long-lived worker
    leaks a task per degraded job.
    """
    s = session()
    client = FakeSTVClient()
    client.close_gate = asyncio.Event()
    s._client = client
    s._audio_output = QueueAudioOutput(sample_rate=24000, wait_playback_start=True)

    degrading = asyncio.create_task(s._degrade())
    await asyncio.sleep(0.01)  # let it suspend inside aclose, on the client close

    await s.aclose()  # the job's shutdown callback, racing the teardown
    client.close_gate.set()
    await asyncio.wait_for(degrading, 2)

    assert s._null_drain_task is None, "a drain was started that nothing will stop"


async def test_aclose_does_not_evict_the_agent_from_the_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The base class removes `avatar_identity` on close.

    For a local-runner avatar that identity is the agent's own participant, so
    running it would end the session the degrade path exists to preserve.
    """
    removed: list[object] = []

    class _RoomApi:
        async def remove_participant(self, req: object) -> None:
            removed.append(req)

    class _JobCtx:
        api = SimpleNamespace(room=_RoomApi())

    monkeypatch.setattr(_avatar_types, "get_job_context", lambda required=False: _JobCtx())

    s = session()
    s._client = FakeSTVClient()
    s._room = SimpleNamespace(  # type: ignore[assignment]
        isconnected=lambda: True,
        name="room",
        local_participant=SimpleNamespace(identity="agent"),
        off=lambda *a: None,
    )

    await s.aclose()

    assert removed == [], "aclose evicted the agent's own participant"


# --- a fatal while a segment is already in flight ---------------------------


async def test_fatal_completes_a_segment_the_runner_already_consumed() -> None:
    """The wedge case: the queue is empty, so only the plugin can finish it.

    The runner takes an utterance's audio *and* its AudioSegmentEnd off the
    queue before the sink decides anything. If a fatal lands while that turn is
    still waiting on the stop edge, nothing is left in the queue for the null
    drain to see, and the session blocks in wait_for_playout() for good.
    """
    s = session()
    s._client = FakeSTVClient()
    sink = _FrameSink()
    s._sink = sink
    audio_output = QueueAudioOutput(sample_rate=24000, wait_playback_start=True)
    s._audio_output = audio_output

    # The session speaks, and the runner drains both items off the queue.
    await audio_output.capture_frame(_frame(100))
    audio_output.flush()
    sink.note_input_segment_open()
    sink.note_input_audio()
    async for item in audio_output:
        if isinstance(item, AudioSegmentEnd):
            sink.note_input_segment_end()
            break
    assert sink.owes_segment_end, "the segment is in flight and only the sink can end it"

    await s._degrade()

    ev = await asyncio.wait_for(audio_output.wait_for_playout(), 2)
    assert ev.interrupted is False
    await s.aclose()


async def test_fatal_reports_a_finished_segment_only_once() -> None:
    """A turn already closed by the stop edge was reported by the runner."""
    s = session()
    s._client = FakeSTVClient()
    sink = _FrameSink()
    s._sink = sink
    audio_output = QueueAudioOutput(sample_rate=24000, wait_playback_start=True)
    s._audio_output = audio_output

    sink.note_input_segment_open()
    sink.note_input_audio()
    await sink.write_audio(make_audio_frame())
    sink.note_input_segment_end()
    sink.on_bot_stopped_speaking()

    # The runner drains the marker and reports it; that is what makes the
    # segment finished rather than merely decided.
    while sink.pending:
        await sink.next_frame()
    assert not sink.owes_segment_end

    reports = 0
    original = audio_output.notify_playback_finished

    def counting(**kwargs: object) -> None:
        nonlocal reports
        reports += 1
        original(**kwargs)  # type: ignore[arg-type]

    audio_output.notify_playback_finished = counting  # type: ignore[method-assign]

    await s._degrade()

    assert reports == 0, "reported a segment the runner had already completed"
    await s.aclose()


# --- startup races ----------------------------------------------------------


async def test_first_frame_wait_is_cut_short_by_a_fatal() -> None:
    """A fatal can land after SESSION_READY, past the gate that would catch it."""
    s = session(first_frame_timeout=30)
    sink = _FrameSink()

    async def fail_soon() -> None:
        await asyncio.sleep(0)
        s._on_fatal("server went away")

    task = asyncio.create_task(fail_soon())
    with pytest.raises(OjinException, match="server went away"):
        await asyncio.wait_for(s._await_first_frame(sink), 2)
    await task


async def test_first_frame_still_times_out_without_a_fatal() -> None:
    s = session(first_frame_timeout=0.05)

    with pytest.raises(OjinException, match="no video frame"):
        await s._await_first_frame(_FrameSink())


async def test_render_deadline_is_polled_on_its_own_interval() -> None:
    """A render timeout below the watchdog must not round up to the watchdog."""
    s = session(watchdog_timeout=15, turn_render_timeout=0.05)
    sink = _FrameSink(turn_render_timeout=0.05)
    s._sink = sink
    sink.note_input_segment_open()
    sink.note_input_audio()
    sink.note_input_segment_end()

    task = asyncio.create_task(s._watchdog())
    try:
        await asyncio.wait_for(_until(lambda: s._render_stalls > 0), 2)
    finally:
        s._closed = True
        await utils.aio.cancel_and_wait(task)


async def _until(predicate) -> None:  # type: ignore[no-untyped-def]
    while not predicate():
        await asyncio.sleep(0.01)


async def test_fatal_completes_a_marker_the_runner_never_drained() -> None:
    """The marker is decided when it is queued, reported when it is drained.

    A fatal in between loses it: the tracker has already cleared its flag, and
    cancelling the runner discards the queue it was sitting in.
    """
    s = session()
    s._client = FakeSTVClient()
    sink = _FrameSink()
    s._sink = sink
    audio_output = QueueAudioOutput(sample_rate=24000, wait_playback_start=True)
    s._audio_output = audio_output

    await audio_output.capture_frame(_frame(100))
    audio_output.flush()
    sink.note_input_segment_open()
    sink.note_input_audio()
    async for item in audio_output:
        if isinstance(item, AudioSegmentEnd):
            sink.note_input_segment_end()
            break
    await sink.write_audio(make_audio_frame())
    sink.on_bot_stopped_speaking()

    assert any(isinstance(f, AudioSegmentEnd) for f in sink.pending), "marker queued"
    assert not sink._segments.owes_segment_end, "and the tracker already let go of it"

    await s._degrade()

    ev = await asyncio.wait_for(audio_output.wait_for_playout(), 2)
    assert ev.interrupted is False
    await s.aclose()


async def test_fatal_between_the_first_frame_and_the_runner_fails_the_start() -> None:
    """_on_fatal cannot schedule a degrade before a runner exists to tear down."""
    s = session()
    s._client = FakeSTVClient()
    sink = _FrameSink()
    s._sink = sink
    agent_session = SimpleNamespace(output=SimpleNamespace(replace_audio_tail=_unreachable))

    real_runner = avatar_mod.AvatarRunner

    class FatalOnStart(real_runner):  # type: ignore[misc, valid-type]
        async def start(self) -> None:
            s._on_fatal("server went away")

        async def aclose(self) -> None:
            return

    avatar_mod.AvatarRunner = FatalOnStart
    try:
        with pytest.raises(OjinException, match="server went away"):
            await s._start_runner(
                agent_session,  # type: ignore[arg-type]
                SimpleNamespace(),  # type: ignore[arg-type]
                sink,
                make_video_frame(),
            )
    finally:
        avatar_mod.AvatarRunner = real_runner
    await s.aclose()


def _unreachable(*_: object, **__: object) -> None:
    raise AssertionError("installed the audio output despite a fatal during startup")
