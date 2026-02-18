from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any

from livekit import rtc
from livekit.browser import AudioData, BrowserPage, PaintData  # type: ignore[import-untyped]

from ._keys import NATIVE_KEY_CODES as _NATIVE_KEY_CODES

logger = logging.getLogger(__name__)


class BrowserSession:
    def __init__(self, *, page: BrowserPage, room: rtc.Room) -> None:
        self._page = page
        self._room = room
        self._video_source: rtc.VideoSource | None = None
        self._audio_source: rtc.AudioSource | None = None
        self._video_track: rtc.LocalVideoTrack | None = None
        self._audio_track: rtc.LocalAudioTrack | None = None
        self._started = False
        self._last_frame: rtc.VideoFrame | None = None
        self._video_task: asyncio.Task[None] | None = None
        self._audio_init_task: asyncio.Task[None] | None = None
        self._audio_task: asyncio.Task[None] | None = None
        self._audio_queue: asyncio.Queue[rtc.AudioFrame] | None = None
        self._input_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=256)
        self._input_task: asyncio.Task[None] | None = None

        self._focus_identity: str | None = None

        # Event set when a human interrupts the agent's focus
        self._agent_interrupted = asyncio.Event()

    @property
    def focus_identity(self) -> str | None:
        return self._focus_identity

    @property
    def agent_interrupted(self) -> asyncio.Event:
        """Event that is set when a human takes focus from the agent."""
        return self._agent_interrupted

    def set_agent_focus(self, active: bool) -> None:
        """Grant or revoke browser focus for the AI agent."""
        if active:
            self._focus_identity = "__agent__"
            self._agent_interrupted.clear()
        elif self._focus_identity == "__agent__":
            self._focus_identity = None

    async def start(self) -> None:
        if self._started:
            return
        self._started = True

        opts = self._page._opts

        self._video_source = rtc.VideoSource(opts.width, opts.height, is_screencast=True)

        self._video_track = rtc.LocalVideoTrack.create_video_track(
            "browser-video", self._video_source
        )

        video_opts = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_SCREENSHARE,
            video_encoding=rtc.VideoEncoding(max_bitrate=8_000_000, max_framerate=opts.framerate),
            simulcast=False,
        )
        self._page.on("paint", self._on_paint)
        self._page.on("audio", self._on_audio)
        self._page.on("cursor_changed", self._on_cursor)
        self._page.on("url_changed", self._on_url_changed)

        await self._room.local_participant.publish_track(self._video_track, video_opts)

        self._video_task = asyncio.create_task(self._video_loop(opts.framerate))

        # Single persistent task for sending input events to subprocess
        self._input_task = asyncio.create_task(self._input_sender_loop())

        # Register RPC methods for focus management
        @self._room.local_participant.register_rpc_method("browser/request-focus")  # type: ignore[arg-type]
        async def _handle_request_focus(
            data: rtc.rpc.RpcInvocationData,
        ) -> str:
            if self._focus_identity is None:
                self._focus_identity = data.caller_identity
                await self._page.send_focus_event(True)
                await self._broadcast_focus()
                return json.dumps({"granted": True})

            # If agent has focus, allow human to interrupt
            if self._focus_identity == "__agent__":
                self._focus_identity = data.caller_identity
                self._agent_interrupted.set()
                await self._page.send_focus_event(True)
                await self._broadcast_focus()
                return json.dumps({"granted": True})

            return json.dumps({"granted": False, "holder": self._focus_identity})

        @self._room.local_participant.register_rpc_method("browser/release-focus")  # type: ignore[arg-type]
        async def _handle_release_focus(
            data: rtc.rpc.RpcInvocationData,
        ) -> str:
            if self._focus_identity == data.caller_identity:
                self._focus_identity = None
                await self._page.send_focus_event(False)
                await self._broadcast_focus()
                return json.dumps({"released": True})
            return json.dumps({"released": False})

        @self._room.local_participant.register_rpc_method("browser/navigate")  # type: ignore[arg-type]
        async def _handle_navigate(
            data: rtc.rpc.RpcInvocationData,
        ) -> str:
            payload = json.loads(data.payload)
            url = payload.get("url", "")
            if url:
                self._queue_input(self._page.navigate(url))
            return json.dumps({"status": "ok"})

        @self._room.local_participant.register_rpc_method("browser/go-back")  # type: ignore[arg-type]
        async def _handle_go_back(
            data: rtc.rpc.RpcInvocationData,
        ) -> str:
            self._queue_input(self._page.go_back())
            return json.dumps({"status": "ok"})

        @self._room.local_participant.register_rpc_method("browser/go-forward")  # type: ignore[arg-type]
        async def _handle_go_forward(
            data: rtc.rpc.RpcInvocationData,
        ) -> str:
            self._queue_input(self._page.go_forward())
            return json.dumps({"status": "ok"})

        # Listen for input data from participants
        @self._room.on("data_received")
        def _on_data_received(packet: rtc.DataPacket) -> None:
            if packet.topic != "browser-input":
                return
            if packet.participant is None:
                return
            if packet.participant.identity != self._focus_identity:
                return
            try:
                events = json.loads(packet.data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
            for evt in events:
                self._dispatch_input(evt)

        self._on_data_received = _on_data_received

        # Release focus when the holder disconnects
        @self._room.on("participant_disconnected")
        def _on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
            if participant.identity == self._focus_identity:
                self._focus_identity = None
                self._queue_input(self._page.send_focus_event(False))
                self._queue_input(self._broadcast_focus())

        self._on_participant_disconnected = _on_participant_disconnected

    def _queue_input(self, coro: Any) -> None:
        try:
            self._input_queue.put_nowait(coro)
        except asyncio.QueueFull:
            pass

    async def _input_sender_loop(self) -> None:
        while True:
            coro = await self._input_queue.get()
            try:
                await coro
            except Exception:
                pass

    def _dispatch_input(self, evt: dict[str, Any]) -> None:
        t = evt.get("type")
        if t == "mousemove":
            self._queue_input(self._page.send_mouse_move(evt["x"], evt["y"]))
        elif t == "mousedown":
            self._queue_input(
                self._page.send_mouse_click(evt["x"], evt["y"], evt.get("button", 0), False, 1)
            )
        elif t == "mouseup":
            self._queue_input(
                self._page.send_mouse_click(evt["x"], evt["y"], evt.get("button", 0), True, 1)
            )
        elif t == "wheel":
            self._queue_input(
                self._page.send_mouse_wheel(
                    evt["x"], evt["y"], evt.get("deltaX", 0), evt.get("deltaY", 0)
                )
            )
        elif t == "keydown":
            wkc = evt["keyCode"]
            nkc = _NATIVE_KEY_CODES.get(wkc, 0)
            self._queue_input(self._page.send_key_event(0, evt.get("modifiers", 0), wkc, nkc, 0))
        elif t == "keyup":
            wkc = evt["keyCode"]
            nkc = _NATIVE_KEY_CODES.get(wkc, 0)
            self._queue_input(self._page.send_key_event(2, evt.get("modifiers", 0), wkc, 0, 0))
        elif t == "char":
            wkc = evt["keyCode"]
            nkc = _NATIVE_KEY_CODES.get(wkc, 0)
            self._queue_input(
                self._page.send_key_event(
                    3, evt.get("modifiers", 0), wkc, nkc, evt.get("charCode", 0)
                )
            )

    async def _broadcast_focus(self) -> None:
        payload = json.dumps({"identity": self._focus_identity}).encode()
        await self._room.local_participant.publish_data(
            payload, reliable=True, topic="browser-focus"
        )

    async def _init_audio(self, frame: rtc.AudioFrame) -> None:
        try:
            self._audio_source = rtc.AudioSource(
                frame.sample_rate, frame.num_channels, queue_size_ms=100
            )
            self._audio_track = rtc.LocalAudioTrack.create_audio_track(
                "browser-audio", self._audio_source
            )
            audio_opts = rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO,
            )
            await self._room.local_participant.publish_track(self._audio_track, audio_opts)
            self._audio_queue = asyncio.Queue(maxsize=50)
            self._audio_queue.put_nowait(frame)
            self._audio_task = asyncio.create_task(self._audio_loop())
        except Exception:
            logger.exception("failed to initialize audio, will retry on next packet")
            self._audio_source = None
            self._audio_track = None
            self._audio_init_task = None

    async def _audio_loop(self) -> None:
        assert self._audio_queue is not None
        assert self._audio_source is not None
        while True:
            frame = await self._audio_queue.get()
            try:
                await self._audio_source.capture_frame(frame)
            except Exception:
                logger.exception("audio capture error")

    def _on_audio(self, data: AudioData) -> None:
        if self._audio_source is None:
            if self._audio_init_task is not None:
                return
            self._audio_init_task = asyncio.get_event_loop().create_task(
                self._init_audio(data.frame)
            )
            return

        if self._audio_queue is None:
            return

        try:
            self._audio_queue.put_nowait(data.frame)
        except asyncio.QueueFull:
            pass

    _CEF_CURSOR_MAP: dict[int, str] = {
        0: "default",  # CT_POINTER
        1: "crosshair",  # CT_CROSS
        2: "pointer",  # CT_HAND
        3: "text",  # CT_IBEAM
        4: "wait",  # CT_WAIT
        5: "help",  # CT_HELP
        6: "ew-resize",  # CT_EASTRESIZE
        7: "ns-resize",  # CT_NORTHRESIZE
        8: "nesw-resize",  # CT_NORTHEASTRESIZE
        9: "nwse-resize",  # CT_NORTHWESTRESIZE
        10: "ns-resize",  # CT_SOUTHRESIZE
        11: "nwse-resize",  # CT_SOUTHEASTRESIZE
        12: "nesw-resize",  # CT_SOUTHWESTRESIZE
        13: "ew-resize",  # CT_WESTRESIZE
        14: "ns-resize",  # CT_NORTHSOUTHRESIZE
        15: "ew-resize",  # CT_EASTWESTRESIZE
        16: "nesw-resize",  # CT_NORTHEASTSOUTHWESTRESIZE
        17: "nwse-resize",  # CT_NORTHWESTSOUTHEASTRESIZE
        18: "ew-resize",  # CT_COLUMNRESIZE
        19: "ns-resize",  # CT_ROWRESIZE
        20: "move",  # CT_MIDDLEPANNING
        28: "not-allowed",  # CT_NOTALLOWED
        29: "grab",  # CT_GRAB
        30: "grabbing",  # CT_GRABBING
        32: "move",  # CT_MOVE
    }

    def _on_url_changed(self, url: str) -> None:
        payload = json.dumps({"url": url}).encode()
        self._queue_input(
            self._room.local_participant.publish_data(payload, reliable=True, topic="browser-url")
        )

    def _on_cursor(self, cursor_type: int) -> None:
        css_cursor = self._CEF_CURSOR_MAP.get(cursor_type, "default")
        payload = json.dumps({"cursor": css_cursor}).encode()
        self._queue_input(
            self._room.local_participant.publish_data(
                payload, reliable=True, topic="cursor_changed"
            )
        )

    def _on_paint(self, data: PaintData) -> None:
        self._last_frame = data.frame

    async def _video_loop(self, fps: float) -> None:
        interval = 1.0 / fps
        loop = asyncio.get_event_loop()
        next_time = loop.time()
        while True:
            if self._last_frame is not None and self._video_source is not None:
                ts_us = time.monotonic_ns() // 1000
                self._video_source.capture_frame(self._last_frame, timestamp_us=ts_us)
            next_time += interval
            delay = next_time - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_time = loop.time()

    async def aclose(self) -> None:
        if not self._started:
            return

        self._page.off("paint", self._on_paint)
        self._page.off("audio", self._on_audio)
        self._page.off("cursor_changed", self._on_cursor)
        self._page.off("url_changed", self._on_url_changed)

        if hasattr(self, "_on_data_received"):
            self._room.off("data_received", self._on_data_received)
        if hasattr(self, "_on_participant_disconnected"):
            self._room.off("participant_disconnected", self._on_participant_disconnected)

        if self._video_task:
            self._video_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._video_task

        if self._audio_task:
            self._audio_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._audio_task

        if self._input_task:
            self._input_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._input_task

        try:
            if self._video_track:
                await self._room.local_participant.unpublish_track(self._video_track.sid)
            if self._audio_track:
                await self._room.local_participant.unpublish_track(self._audio_track.sid)
        except Exception:
            pass  # tracks may already be gone if room disconnected first
