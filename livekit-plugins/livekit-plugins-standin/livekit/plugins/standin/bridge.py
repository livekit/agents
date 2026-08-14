# Copyright 2026 Komaa DigiTech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The embedded call listener: what makes this plugin standalone.

StandIn's media bridge dials ``wss://<your-host>/msteams/calling/{callId}`` per
call, exactly as it always has. This listener answers that dial inside the
worker process - no separate bridge to run, nothing to change on StandIn's
side. Per call it creates one LiveKit room, dispatches the worker's own agent
into it by name, publishes the caller's audio as a room track, and relays the
agent's audio back to Teams.

Defaults match the agreed livekit layout: port **8080**, path
``/msteams/calling``. Expose the port (for example
``tailscale funnel --bg --set-path /msteams/calling http://127.0.0.1:8080/msteams/calling``)
and register the public ``wss://`` URL as the identity's agent voice URL.

The room is the hand-off to the entrypoint: the dispatched job sees the caller
as an ordinary audio track, reads :class:`~.call.CallInfo` from the dispatch
metadata, and gets call context and the governor's goodbye on the
``msteams.context`` / ``msteams.goodbye`` data topics - the same contract
:class:`~.call.TeamsCall` consumes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from datetime import timedelta
from typing import Any
from urllib.parse import urlparse, urlunparse

from aiohttp import WSMsgType, web

from livekit import api, rtc

from ._exceptions import StandInError
from ._hmac import (
    REPLAY_WINDOW_MS,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    now_ms,
    verify_handshake,
)
from .call import TOPIC_CONTEXT, TOPIC_GOODBYE
from .log import logger
from .protocol import (
    NUM_CHANNELS,
    SAMPLE_RATE_HZ,
    SessionStart,
    audio_frame,
    decode_pcm,
    parse_message,
    parse_session_start,
    pong,
    session_end,
)

#: callId reaches us as a decoded URL segment, so it can contain anything a
#: %-escape can smuggle. Room names get a conservative charset.
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")

#: Log-safe rendering of an attacker-influenceable id: control characters
#: (CR/LF forge log lines) replaced, length bounded.
_CTRL = re.compile(r"[^\x20-\x7e]")


def _safe(value: str) -> str:
    return _CTRL.sub("?", value)[:80]


_BRIDGE_IDENTITY = "standin-bridge"

#: How often the single-use handshake cache is swept for expired entries.
_PRUNE_INTERVAL_MS = 1_000


def _http_url(url: str) -> str:
    """LiveKitAPI speaks HTTP; accept the ws(s):// form people configure."""
    parts = urlparse(url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    return urlunparse(parts._replace(scheme=scheme))


class _Call:
    """One live call: the StandIn socket on one side, a LiveKit room on the other."""

    def __init__(self, bridge: CallBridge, call_id: str, ws: web.WebSocketResponse) -> None:
        self._bridge = bridge
        self._call_id = call_id
        self._ws = ws
        self._room: rtc.Room | None = None
        self._room_name = ""
        self._source: rtc.AudioSource | None = None
        self._audio_stream: rtc.AudioStream | None = None
        self._agent_identity: str | None = None
        self._pump_sid: str | None = None
        self._seq = 0
        self._sent_ms = 0
        self._last_audio: float | None = None
        self._closed = False
        self._closing_reason = "call-ended"
        self._close_task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        #: Context published before the agent's audio binds would reach nobody:
        #: the agent needs seconds to join after dispatch, and a data packet
        #: reaches only participants connected at that instant. Queue until the
        #: agent is bound, then flush.
        self._pending_context: list[tuple[str, str]] = []

    # ---- lifecycle ----

    async def run(self) -> None:
        """Drive the socket until it ends. Never raises to the caller: a single
        bad call must not take the worker with it."""
        try:
            await self._pump_worker()
        except asyncio.CancelledError:
            self._begin_close("bridge-shutdown")
            raise
        except Exception:
            logger.exception("standin: call %s failed", _safe(self._call_id))
            self._begin_close("transport-failure")
        finally:
            await self.aclose()

    async def _pump_worker(self) -> None:
        started = False
        async for msg in self._ws:
            if msg.type is WSMsgType.ERROR:
                raise StandInError(f"call socket error: {self._ws.exception()}")
            if msg.type is not WSMsgType.TEXT:
                continue
            frame = parse_message(msg.data)
            if frame is None:
                continue
            kind = frame["type"]
            if kind == "session.start":
                if started:
                    continue  # a second start is a sender bug, not a new call
                started = True
                await self._on_session_start(parse_session_start(frame))
            elif kind == "audio.frame":
                self._last_audio = time.monotonic()
                await self._on_caller_audio(frame)
            elif kind == "ping":
                await self._send(pong(frame.get("ts")))
            elif kind == "participants":
                # The same sentences the standalone bridges publish, so agents
                # written against either read identical context.
                count = frame.get("count")
                if isinstance(count, int):
                    if count <= 1:
                        sentence = "This is a 1:1 call with a single human caller."
                    else:
                        sentence = (
                            f"There are {count} human participants on this call. "
                            "Stay quiet unless directly addressed."
                        )
                    await self._publish_context(sentence)
            elif kind == "dtmf":
                digit = frame.get("digit")
                if isinstance(digit, str) and digit:
                    await self._publish_context(
                        f'The caller pressed the "{digit}" key on their keypad.'
                    )
            elif kind == "recording.status":
                status = frame.get("status")
                if isinstance(status, str):
                    await self._publish_context(
                        "The Microsoft Teams call recording is now ACTIVE."
                        if status == "active"
                        else "The Microsoft Teams call recording is not active."
                    )
            elif kind == "assistant.say":
                text = frame.get("text")
                if isinstance(text, str) and text.strip():
                    # Flush the worker's queued agent playback FIRST: without the
                    # cancel, the goodbye publishes behind seconds of already-
                    # buffered audio and the call is torn down before it plays.
                    await self._send(
                        json.dumps(
                            {"type": "assistant.cancel", "turnId": self._seq},
                            separators=(",", ":"),
                        )
                    )
                    await self._publish_text(TOPIC_GOODBYE, text)
            elif kind == "session.end":
                self._closing_reason = str(frame.get("reason") or "call-ended")
                return
            # Anything else (the avatar surface included) is ignored by
            # contract, so an older plugin and a newer StandIn interoperate.

    async def _on_session_start(self, start: SessionStart) -> None:
        if start.call_id != self._call_id:
            # The URL path is what the HMAC signed. A body that disagrees is
            # either a bug or an attempt to ride one call's signature into
            # another's room.
            raise StandInError(
                f"session.start callId {start.call_id!r} does not match the authenticated path"
            )
        await self._join_room(start)

    def _begin_close(self, reason: str) -> asyncio.Task[None]:
        """Start (or return the in-flight) teardown task. Idempotent, and the
        FIRST reason wins - a cascade of close causes must not overwrite the
        one that actually ended the call."""
        if self._close_task is None:
            self._closing_reason = reason
            self._closed = True
            self._close_task = asyncio.ensure_future(self._teardown())
        return self._close_task

    async def aclose(self, reason: str | None = None) -> None:
        """Every caller waits for the SAME teardown, shielded: a caller being
        cancelled (the pre-start watchdog, a task in _tasks that teardown
        itself cancels, worker shutdown) must never abort teardown mid-flight -
        that is how a slot leaks and a callId 409s forever."""
        task = self._begin_close(reason or self._closing_reason)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            # Only the CALLER was cancelled; teardown continues on its own.
            raise
        except Exception:
            pass  # teardown already logged; never raise into callers

    async def _teardown(self) -> None:
        try:
            # Teardown runs in its OWN task, never in self._tasks, so cancelling
            # the whole set is safe here.
            for task in list(self._tasks):
                task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

            # rtc primitives own FFI subscriptions and internal tasks that only
            # aclose() releases: cancelling the tasks above frees neither, so
            # every finished call would leave one of each behind for the life of
            # the worker. Closed here, before the socket and room teardown, so a
            # hang there cannot skip them.
            stream, self._audio_stream = self._audio_stream, None
            if stream is not None:
                with contextlib.suppress(Exception):
                    await stream.aclose()
            source, self._source = self._source, None
            if source is not None:
                with contextlib.suppress(Exception):
                    await source.aclose()

            with contextlib.suppress(Exception):
                if not self._ws.closed:
                    await self._send(session_end(self._closing_reason))
                    await self._ws.close()

            room, self._room = self._room, None
            if room is not None:
                with contextlib.suppress(Exception):
                    await room.disconnect()
                await self._delete_room()
        finally:
            # Unconditional: whatever failed or was cancelled above, the slot is
            # released and the callId becomes usable again.
            self._bridge._release(self._call_id)
            logger.info("standin: call %s ended (%s)", _safe(self._call_id), self._closing_reason)

    # ---- LiveKit side ----

    async def _join_room(self, start: SessionStart) -> None:
        bridge = self._bridge
        # Same 100-char budget the standalone bridges use, so both derive the
        # same room name for the same call.
        self._room_name = f"{bridge.room_prefix}{_UNSAFE.sub('-', self._call_id)}"[:100]

        token = (
            api.AccessToken(bridge._api_key, bridge._api_secret)
            .with_identity(_BRIDGE_IDENTITY)
            .with_name("Microsoft Teams")
            .with_ttl(timedelta(hours=6))
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=self._room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .to_jwt()
        )

        room = rtc.Room()
        self._room = room
        self._wire_room_events(room)
        await room.connect(bridge._livekit_url, token, rtc.RoomOptions(auto_subscribe=True))
        logger.info('standin: call %s joined room "%s"', self._call_id, self._room_name)

        # Dispatch AFTER connect, because connect is what creates the room.
        await self._dispatch_agent(start)

        source = rtc.AudioSource(SAMPLE_RATE_HZ, NUM_CHANNELS)
        track = rtc.LocalAudioTrack.create_audio_track("teams-caller", source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await room.local_participant.publish_track(track, options)
        self._source = source
        self._last_audio = time.monotonic()
        self._spawn(self._watch_audio_idle())

    async def _watch_audio_idle(self) -> None:
        """End the call when the caller's audio stops arriving.

        A live Teams call delivers PCM continuously - silence is still frames -
        so audio going quiet for this long means the call is gone on the far
        side and nobody told us. That happens: the peer keeps the socket open
        (and even keeps pinging) while its own teardown is wedged, and without
        this backstop the room, the agent job, and the realtime session all burn
        until someone notices."""
        idle = self._bridge.audio_idle_timeout
        if idle <= 0:
            return
        while not self._closed:
            await asyncio.sleep(min(idle / 4, 10.0))
            last = self._last_audio
            if last is not None and time.monotonic() - last > idle:
                logger.warning(
                    "standin: call %s got no caller audio for %.0fs; ending it",
                    self._call_id,
                    idle,
                )
                self._closing_reason = "caller-idle-timeout"
                await self.aclose()
                return

    async def _dispatch_agent(self, start: SessionStart) -> None:
        bridge = self._bridge
        if not bridge.agent_name:
            return  # automatic dispatch projects; nothing to do
        # The same metadata shape the deployed livekit bridges send, so CallInfo
        # and existing agents read it unchanged; call_id/thread_id are additive.
        metadata = {
            "source": "msteams",
            "caller_name": start.caller.display_name or "caller",
            "tenant_id": start.tenant_id or start.caller.tenant_id or "unknown-tenant",
            "call_direction": start.direction,
            "call_id": self._call_id,
            "thread_id": start.thread_id,
        }
        if start.caller.aad_id:
            metadata["user_id"] = start.caller.aad_id
        lkapi = api.LiveKitAPI(_http_url(bridge._livekit_url), bridge._api_key, bridge._api_secret)
        try:
            await lkapi.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=bridge.agent_name,
                    room=self._room_name,
                    metadata=json.dumps(metadata),
                )
            )
            logger.info('standin: dispatched "%s" into %s', bridge.agent_name, self._room_name)
        finally:
            await lkapi.aclose()

    async def _delete_room(self) -> None:
        """Delete the room so the agent job ends at once instead of idling out."""
        bridge = self._bridge
        if not bridge.delete_room_on_end or not self._room_name:
            return
        lkapi = api.LiveKitAPI(_http_url(bridge._livekit_url), bridge._api_key, bridge._api_secret)
        try:
            await lkapi.room.delete_room(api.DeleteRoomRequest(room=self._room_name))
        except Exception as err:
            logger.warning("standin: delete_room failed, room will idle out: %s", err)
        finally:
            with contextlib.suppress(Exception):
                await lkapi.aclose()

    def _wire_room_events(self, room: rtc.Room) -> None:
        @room.on("track_subscribed")
        def _on_track(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            # Bind the agent by participant KIND, not by whoever publishes audio
            # first: in a room with a recorder or a monitor, first-audio-wins
            # binds the wrong identity and then blocks the real agent behind the
            # single-pump gate.
            if self._agent_identity is None:
                if self._bridge.agent_name and not _is_agent(participant):
                    logger.debug('standin: ignoring audio from "%s"', participant.identity)
                    return
                self._agent_identity = participant.identity
                # The agent can hear us now: deliver the context that arrived
                # while the room had nobody to deliver it to.
                self._flush_pending_context()
            elif participant.identity != self._agent_identity:
                return
            self._start_pump(track)

        @room.on("track_unsubscribed")
        def _on_unsubscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if track.sid and track.sid == self._pump_sid:
                self._pump_sid = None  # a re-published track may take over

        @room.on("participant_disconnected")
        def _on_left(participant: rtc.RemoteParticipant) -> None:
            if self._agent_identity and participant.identity == self._agent_identity:
                self._end_soon("agent-disconnected")

        @room.on("disconnected")
        def _on_disconnected(*_: Any) -> None:
            # Final by the time it fires: the SDK retries transient drops first.
            self._end_soon("room-disconnected")

    def _start_pump(self, track: rtc.Track) -> None:
        """Relay the agent's audio back to Teams. One voice at a time."""
        if self._pump_sid:
            return
        sid = track.sid or "unknown"
        self._pump_sid = sid

        async def pump() -> None:
            stream: rtc.AudioStream | None = None
            try:
                # Ask the SDK for 16 kHz mono so our side stays copy-only.
                stream = rtc.AudioStream.from_track(
                    track=track, sample_rate=SAMPLE_RATE_HZ, num_channels=NUM_CHANNELS
                )
                self._audio_stream = stream
                async for event in stream:
                    if self._closed:
                        break
                    pcm = event.frame.data.tobytes()
                    self._seq += 1
                    # The timeline persists across pumps: a re-published track
                    # (avatar swap, mute cycle) must not jump timestampMs back
                    # to zero while seq keeps climbing.
                    await self._send(audio_frame(self._seq, self._sent_ms, pcm))
                    self._sent_ms += (len(pcm) // 2) * 1000 // SAMPLE_RATE_HZ
            except asyncio.CancelledError:
                raise
            except Exception:
                if not self._closed:
                    logger.exception("standin: agent audio pump failed")
            finally:
                # Release the claim only if it is still OURS: a stale pump
                # draining its stream after a takeover must not clear the new
                # pump's claim and let a third pump start alongside it.
                if self._pump_sid == sid:
                    self._pump_sid = None
                if stream is not None:
                    if self._audio_stream is stream:
                        self._audio_stream = None
                    # The stream owns an FFI subscription and an internal task
                    # that only aclose() releases; ending the pump does not.
                    with contextlib.suppress(Exception):
                        await stream.aclose()

        self._spawn(pump())

    # ---- StandIn side ----

    async def _on_caller_audio(self, frame: dict[str, Any]) -> None:
        source = self._source
        if source is None or self._closed:
            return  # audio before session.start, or after teardown
        try:
            pcm = decode_pcm(frame.get("payloadBase64"))
        except ValueError as err:
            logger.warning("standin: dropping caller frame: %s", err)
            return
        await source.capture_frame(
            rtc.AudioFrame(
                data=pcm,
                sample_rate=SAMPLE_RATE_HZ,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(pcm) // 2,
            )
        )

    # ---- plumbing ----

    async def _publish_context(self, text: str) -> None:
        """Context is queued until the agent's audio is bound: a data packet
        reaches only participants connected at that instant, and the initial
        'participants' arrives seconds before the dispatched agent joins."""
        if self._agent_identity is None:
            self._pending_context.append((TOPIC_CONTEXT, text))
            del self._pending_context[:-16]
            return
        await self._publish_text(TOPIC_CONTEXT, text)

    def _flush_pending_context(self) -> None:
        pending, self._pending_context = self._pending_context, []
        for topic, text in pending:
            self._spawn(self._publish_text(topic, text))

    async def _publish_text(self, topic: str, text: str) -> None:
        """The topic contract: both topics carry ``{"text": ...}``."""
        room = self._room
        if room is None or self._closed:
            return
        with contextlib.suppress(Exception):
            await room.local_participant.publish_data(
                json.dumps({"text": text}).encode("utf-8"), reliable=True, topic=topic
            )

    async def _send(self, text: str) -> None:
        if self._ws.closed:
            return
        with contextlib.suppress(Exception):
            await self._ws.send_str(text)

    def _end_soon(self, reason: str) -> None:
        """Close from a synchronous room callback."""
        self._begin_close(reason)

    def _spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


def _is_agent(participant: rtc.RemoteParticipant) -> bool:
    kind = getattr(participant, "kind", None)
    expected = getattr(rtc.ParticipantKind, "PARTICIPANT_KIND_AGENT", None)
    # Assume agent when the SDK reports no kind, so an older rtc build degrades
    # to first-audio-wins rather than never binding.
    return kind is None or expected is None or bool(kind == expected)


class CallBridge:
    """The listener StandIn dials. Started by :func:`~.service.serve`; construct
    it directly only when embedding without an AgentServer.

    Args:
        secret: the connection secret from the StandIn portal. Must byte-match,
            or the handshake is rejected with 401. Defaults to ``STANDIN_SECRET``.
        agent_name: filled in by ``serve()`` from the worker's own registration;
            set explicitly only when embedding.
        livekit_url / livekit_api_key / livekit_api_secret: your LiveKit
            project, defaulting to the standard env variables.
        host / port / ws_path: where the listener binds. Defaults to the agreed
            livekit layout, ``0.0.0.0:8080`` at ``/msteams/calling``.
        room_prefix: room names are ``{room_prefix}{callId}``.
        delete_room_on_end: delete the room at teardown so the agent job ends
            immediately.
        max_connections: concurrent live calls, checked before any crypto runs.
        pre_start_timeout: seconds a socket may stay silent after authenticating
            before it is dropped for never sending ``session.start``.
        audio_idle_timeout: seconds without caller audio before a live call is
            declared dead and torn down (0 disables). A live Teams call streams
            PCM continuously, silence included, so this fires only when the far
            side is gone but its socket was never closed.
    """

    def __init__(
        self,
        *,
        secret: str | None = None,
        agent_name: str = "",
        livekit_url: str | None = None,
        livekit_api_key: str | None = None,
        livekit_api_secret: str | None = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        ws_path: str = "/msteams/calling",
        room_prefix: str = "msteams-",
        delete_room_on_end: bool = True,
        max_connections: int = 64,
        pre_start_timeout: float = 10.0,
        audio_idle_timeout: float = 45.0,
    ) -> None:
        self._secret = secret or os.environ.get("STANDIN_SECRET", "")
        if not self._secret:
            raise StandInError(
                "a StandIn connection secret is required: pass secret=... or set STANDIN_SECRET"
            )
        self.agent_name = agent_name
        self._livekit_url = livekit_url or os.environ.get("LIVEKIT_URL", "")
        self._api_key = livekit_api_key or os.environ.get("LIVEKIT_API_KEY", "")
        self._api_secret = livekit_api_secret or os.environ.get("LIVEKIT_API_SECRET", "")
        missing = [
            name
            for name, value in (
                ("LIVEKIT_URL", self._livekit_url),
                ("LIVEKIT_API_KEY", self._api_key),
                ("LIVEKIT_API_SECRET", self._api_secret),
            )
            if not value
        ]
        if missing:
            raise StandInError(f"LiveKit project not configured: missing {', '.join(missing)}")

        self._host = host
        self._port = port
        self._ws_path = "/" + (ws_path or "").strip().strip("/")
        if self._ws_path == "/":
            raise StandInError("ws_path must be a real path such as /msteams/calling")
        self.room_prefix = room_prefix
        self.delete_room_on_end = delete_room_on_end
        self._max_connections = max_connections
        self._pre_start_timeout = pre_start_timeout
        #: Seconds without caller audio before a live call is declared dead
        #: (0 disables). A live Teams call streams PCM continuously, silence
        #: included, so this only fires when the far side is gone or wedged.
        self.audio_idle_timeout = audio_idle_timeout

        self._calls: dict[str, _Call] = {}
        #: fingerprint -> signing timestamp (ms). Pruned by AGE, never wholesale:
        #: clearing the set would reopen the replay window for every handshake
        #: still inside it.
        self._used_signatures: dict[str, int] = {}
        self._last_prune = now_ms()
        self.draining = False
        self._runner: web.AppRunner | None = None

    @property
    def ws_path(self) -> str:
        return self._ws_path

    @property
    def active_calls(self) -> int:
        return len(self._calls)

    async def start(self) -> None:
        """Bind the listener. Transactional: either it is listening when this
        returns, or nothing of it survives."""
        app = web.Application()
        app.router.add_get("/healthz", self._healthz)
        app.router.add_get(f"{self._ws_path}/{{call_id}}", self._upgrade)

        runner = web.AppRunner(app)
        await runner.setup()
        try:
            await web.TCPSite(runner, self._host, self._port).start()
        except BaseException:
            with contextlib.suppress(Exception):
                await runner.cleanup()
            raise
        self._runner = runner
        logger.info(
            "standin: answering Teams calls on %s:%s%s", self._host, self._port, self._ws_path
        )

    async def aclose(self) -> None:
        """Drain live calls, then stop listening. Awaits the calls' REAL
        teardown tasks - an aclose that early-returns on an in-flight closer
        would let the loop stop with teardown still pending, leaking rooms."""
        calls = list(self._calls.values())
        if calls:
            await asyncio.gather(
                *(c._begin_close("bridge-shutdown") for c in calls), return_exceptions=True
            )
        self._calls.clear()
        runner, self._runner = self._runner, None
        if runner is not None:
            with contextlib.suppress(Exception):
                await runner.cleanup()

    # ---- request handling ----

    async def _healthz(self, _: web.Request) -> web.Response:
        return web.json_response({"ok": True, "calls": len(self._calls)})

    async def _upgrade(self, request: web.Request) -> web.StreamResponse:
        call_id = request.match_info.get("call_id", "")
        if not call_id:
            return web.Response(status=400, text="missing callId")

        # Draining: live calls continue, new ones are refused so a worker that
        # is winding down does not accept calls it will never dispatch.
        if self.draining:
            return web.Response(status=503, text="draining")

        # Capacity is checked BEFORE any crypto, so a flood cannot make us spend
        # CPU on signatures for calls we were never going to accept.
        if len(self._calls) >= self._max_connections:
            logger.warning("standin: refusing %s, at capacity", _safe(call_id))
            return web.Response(status=503, text="at capacity")

        timestamp = request.headers.get(TIMESTAMP_HEADER)
        signature = request.headers.get(SIGNATURE_HEADER)
        if not verify_handshake(self._secret, timestamp, call_id, signature):
            return web.Response(status=401, text="unauthorized")

        # Single-use handshake: a correctly signed upgrade replayed inside the
        # freshness window must not open a second socket. The fingerprint uses
        # the NORMALIZED signature - verify accepts case/whitespace variants, so
        # keying on the raw header would let the same capture replay once per
        # casing. Entries are pruned by age, matching the freshness window.
        sig_norm = (signature or "").strip().lower()
        fingerprint = f"{timestamp}.{sig_norm}"
        now = now_ms()
        if fingerprint in self._used_signatures:
            return web.Response(status=401, text="handshake already used")
        # Key on the SIGNING timestamp, never the arrival time: verification
        # accepts a timestamp up to REPLAY_WINDOW_MS in the FUTURE, so an entry
        # aged from arrival can be pruned while its signature is still valid -
        # reopening the exact replay this guard exists to close. Aged from the
        # signing time, an entry lives precisely as long as the signature does.
        self._used_signatures[fingerprint] = int(timestamp) if timestamp else now
        # Prune on a time throttle, not on a size threshold: rebuilding once the
        # map passes a watermark makes every later request O(n). Only correctly
        # signed, not-yet-seen handshakes reach this line (a bad signature 401s
        # earlier, a replay returns above), so the map tracks StandIn's real
        # call rate rather than attacker traffic.
        if now - self._last_prune >= _PRUNE_INTERVAL_MS:
            self._last_prune = now
            cutoff = now - REPLAY_WINDOW_MS
            self._used_signatures = {
                fp: ts for fp, ts in self._used_signatures.items() if ts >= cutoff
            }

        if call_id in self._calls:
            return web.Response(status=409, text="call already has a live session")

        # 2 MB bounds a single inbound message, matching the sibling providers:
        # audio is ~856 B base64 per frame, and the protocol caps video.frame
        # JPEGs to fit this envelope (sent sparsely, dropped when busy).
        ws = web.WebSocketResponse(heartbeat=None, max_msg_size=2 * 1024 * 1024)
        await ws.prepare(request)

        call = _Call(self, call_id, ws)
        self._calls[call_id] = call
        logger.info("standin: call %s connected", _safe(call_id))

        watchdog = asyncio.ensure_future(self._watch_pre_start(call))
        try:
            await call.run()
        finally:
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog
        return ws

    async def _watch_pre_start(self, call: _Call) -> None:
        """Drop a socket that authenticates and then never starts a call: it
        holds a connection slot and a callId that nothing will ever free."""
        await asyncio.sleep(self._pre_start_timeout)
        if call._room is None and not call._closed:
            logger.warning("standin: call %s never sent session.start", _safe(call._call_id))
            await call.aclose("pre-start-timeout")

    def _release(self, call_id: str) -> None:
        self._calls.pop(call_id, None)


__all__ = ["CallBridge"]
