"""Pinch real-time speech-to-speech translation plugin for LiveKit Agents."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import Callable

from livekit import rtc

from .log import logger
from .models import TranscriptEvent, TranslatorOptions

_SAMPLE_RATE = 48_000
_NUM_CHANNELS = 1

_MAX_CONNECT_RETRIES = 3
_RETRY_BASE_DELAY = 1.0

_AGENT_TRACK_TIMEOUT = 30.0


class PinchError(Exception):
    """Base class for all Pinch plugin errors."""


class PinchAuthError(PinchError):
    """Raised when the API key is invalid or missing (HTTP 401)."""


class PinchRateLimitError(PinchError):
    """Raised when the API rate limit is exceeded (HTTP 429)."""


class PinchSessionError(PinchError):
    """Raised when session creation or room connection fails."""


class Translator:
    """Real-time speech-to-speech translation via the Pinch API.

    Usage::

        from livekit.plugins.pinch import Translator, TranslatorOptions

        translator = Translator(
            options=TranslatorOptions(
                source_language="en-US",
                target_language="es-ES",
                voice_type="clone",
            )
        )

        @translator.on_transcript
        def handle(event):
            if event.is_translated and event.is_final:
                print(f"[{event.language_detected}] {event.text}")

        await translator.start(room)
        await translator.stop()

    Parameters
    ----------
    api_key:
        Pinch API key. Falls back to the ``PINCH_API_KEY`` environment variable.
    options:
        Source/target languages and voice type.
    """

    def __init__(
        self,
        *,
        options: TranslatorOptions,
        api_key: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("PINCH_API_KEY")
        if not resolved_key:
            raise ValueError(
                "A Pinch API key is required. Supply it via the `api_key` "
                "argument or set the PINCH_API_KEY environment variable."
            )
        self._api_key: str = resolved_key
        self._options: TranslatorOptions = options

        self._source_room: rtc.Room | None = None
        self._pinch_room: rtc.Room | None = None

        self._pinch_audio_source: rtc.AudioSource | None = None
        self._source_audio_source: rtc.AudioSource | None = None

        self._pinch_local_track: rtc.LocalAudioTrack | None = None
        self._source_local_track: rtc.LocalAudioTrack | None = None

        self._pinch_publication: rtc.LocalTrackPublication | None = None
        self._source_publication: rtc.LocalTrackPublication | None = None

        self._tasks: list[asyncio.Task] = []
        self._transcript_callbacks: list[Callable[[TranscriptEvent], None]] = []

        self._started = False
        self._stopped = False

    def on_transcript(
        self, callback: Callable[[TranscriptEvent], None]
    ) -> Callable[[TranscriptEvent], None]:
        """Register a transcript callback. Works as a decorator or direct call.

        Called for both ``original_transcript`` and ``translated_transcript`` events.
        """
        self._transcript_callbacks.append(callback)
        return callback

    def remove_transcript_listener(self, callback: Callable[[TranscriptEvent], None]) -> None:
        """Remove a previously registered transcript callback."""
        with contextlib.suppress(ValueError):
            self._transcript_callbacks.remove(callback)

    async def start(self, source_room: rtc.Room) -> None:
        """Start the Pinch translation pipeline.

        Raises
        ------
        PinchAuthError
            If the API key is rejected.
        PinchRateLimitError
            If the API rate limit is exceeded.
        PinchSessionError
            If session creation or room connection fails after retries.
        RuntimeError
            If called after :meth:`stop`.
        """
        if self._stopped:
            raise RuntimeError(
                "This Translator has already been stopped. "
                "Create a new instance to start a fresh session."
            )
        if self._started:
            logger.warning("Translator.start() called more than once; ignoring.")
            return

        self._started = True
        self._source_room = source_room

        logger.info(
            "Pinch Translator starting  src=%s  tgt=%s  voice=%s",
            self._options.source_language,
            self._options.target_language,
            self._options.voice_type,
        )

        session = await self._create_session()
        pinch_url: str = session["url"]
        pinch_token: str = session["token"]
        pinch_room_name: str = session["room_name"]
        logger.info("Pinch session created  room=%s", pinch_room_name)

        self._pinch_room = await self._connect_with_retry(pinch_url, pinch_token)
        logger.info("Connected to Pinch room  room=%s", pinch_room_name)

        self._pinch_audio_source = rtc.AudioSource(
            sample_rate=_SAMPLE_RATE, num_channels=_NUM_CHANNELS
        )
        self._pinch_local_track = rtc.LocalAudioTrack.create_audio_track(
            "pinch-input", self._pinch_audio_source
        )
        self._pinch_publication = await self._pinch_room.local_participant.publish_track(
            self._pinch_local_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

        self._source_audio_source = rtc.AudioSource(
            sample_rate=_SAMPLE_RATE, num_channels=_NUM_CHANNELS
        )
        self._source_local_track = rtc.LocalAudioTrack.create_audio_track(
            "pinch-translated", self._source_audio_source
        )
        self._source_publication = await self._source_room.local_participant.publish_track(
            self._source_local_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

        self._pinch_room.on("data_received", self._on_pinch_data_received)

        self._tasks.append(asyncio.ensure_future(self._run_source_to_pinch_bridge()))
        self._tasks.append(asyncio.ensure_future(self._run_pinch_to_source_bridge()))

        logger.info("Pinch Translator started successfully.")

    async def stop(self) -> None:
        """Shut down the translation pipeline. Safe to call multiple times."""
        if self._stopped:
            return
        self._stopped = True
        logger.info("Stopping Pinch Translator…")

        if self._pinch_room is not None:
            with contextlib.suppress(Exception):
                self._pinch_room.off("data_received", self._on_pinch_data_received)

        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self._pinch_room is not None:
            if self._pinch_publication is not None:
                with contextlib.suppress(Exception):
                    await self._pinch_room.local_participant.unpublish_track(
                        self._pinch_publication.sid
                    )
            with contextlib.suppress(Exception):
                await self._pinch_room.disconnect()
            self._pinch_room = None

        if self._source_room is not None and self._source_publication is not None:
            with contextlib.suppress(Exception):
                await self._source_room.local_participant.unpublish_track(
                    self._source_publication.sid
                )

        self._pinch_audio_source = None
        self._source_audio_source = None

        logger.info("Pinch Translator stopped.")

    async def _create_session(self) -> dict:
        """Create a Pinch translation session via the SDK."""
        from pinch import PinchClient, SessionParams
        from pinch.errors import (
            PinchAuthError as _SdkAuthError,
            PinchError as _SdkError,
            PinchRateLimitError as _SdkRateLimitError,
        )

        params = SessionParams(
            source_language=self._options.source_language,
            target_language=self._options.target_language,
            voice_type=self._options.voice_type,
        )
        try:
            client = PinchClient(api_key=self._api_key)
            session_info = await asyncio.to_thread(client.create_session, params)
        except _SdkAuthError as exc:
            raise PinchAuthError(str(exc)) from exc
        except _SdkRateLimitError as exc:
            raise PinchRateLimitError(str(exc)) from exc
        except _SdkError as exc:
            raise PinchSessionError(str(exc)) from exc

        return {
            "url": session_info.url,
            "token": session_info.token,
            "room_name": session_info.room_name,
        }

    async def _connect_with_retry(self, url: str, token: str) -> rtc.Room:
        last_exc: Exception | None = None

        for attempt in range(_MAX_CONNECT_RETRIES):
            room = rtc.Room()
            try:
                await room.connect(url, token)
                return room
            except Exception as exc:
                last_exc = exc
                if attempt == _MAX_CONNECT_RETRIES - 1:
                    break
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "Connection attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1,
                    _MAX_CONNECT_RETRIES,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        raise PinchSessionError(
            f"Could not connect to the Pinch room after {_MAX_CONNECT_RETRIES} attempts."
        ) from last_exc

    async def _run_source_to_pinch_bridge(self) -> None:
        if self._source_room is None:
            return

        streaming_tasks: list[asyncio.Task] = []

        async def _stream_track(track: rtc.RemoteAudioTrack) -> None:
            try:
                audio_stream = rtc.AudioStream(
                    track, sample_rate=_SAMPLE_RATE, num_channels=_NUM_CHANNELS
                )
                async for event in audio_stream:
                    if self._stopped:
                        break
                    if self._pinch_audio_source is not None:
                        await self._pinch_audio_source.capture_frame(event.frame)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if not self._stopped:
                    logger.warning("source→Pinch bridge error on track %s: %s", track.sid, exc)

        def start_streaming(track: rtc.Track) -> None:
            if isinstance(track, rtc.RemoteAudioTrack):
                streaming_tasks.append(asyncio.ensure_future(_stream_track(track)))

        for participant in self._source_room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track is not None:
                    start_streaming(pub.track)

        @self._source_room.on("track_subscribed")
        def _on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            start_streaming(track)

        try:
            while not self._stopped:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            self._source_room.off("track_subscribed", _on_track_subscribed)
            for t in streaming_tasks:
                t.cancel()
            if streaming_tasks:
                await asyncio.gather(*streaming_tasks, return_exceptions=True)

    async def _run_pinch_to_source_bridge(self) -> None:
        if self._pinch_room is None:
            return

        streaming_tasks: list[asyncio.Task] = []

        async def _stream_track(track: rtc.RemoteAudioTrack) -> None:
            try:
                audio_stream = rtc.AudioStream(
                    track, sample_rate=_SAMPLE_RATE, num_channels=_NUM_CHANNELS
                )
                async for event in audio_stream:
                    if self._stopped:
                        break
                    if self._source_audio_source is not None:
                        await self._source_audio_source.capture_frame(event.frame)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if not self._stopped:
                    logger.warning("Pinch→source bridge error on track %s: %s", track.sid, exc)

        def start_streaming(track: rtc.Track) -> None:
            if isinstance(track, rtc.RemoteAudioTrack):
                logger.info(
                    "Pinch agent published audio track sid=%s — forwarding to source room.",
                    track.sid,
                )
                streaming_tasks.append(asyncio.ensure_future(_stream_track(track)))

        for participant in self._pinch_room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track is not None:
                    start_streaming(pub.track)

        if not any(
            isinstance(pub.track, rtc.RemoteAudioTrack)
            for p in self._pinch_room.remote_participants.values()
            for pub in p.track_publications.values()
        ):
            logger.info(
                "Waiting for Pinch translation agent to publish audio (timeout=%.0fs)…",
                _AGENT_TRACK_TIMEOUT,
            )

        @self._pinch_room.on("track_subscribed")
        def _on_pinch_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            start_streaming(track)

        try:
            while not self._stopped:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            self._pinch_room.off("track_subscribed", _on_pinch_track_subscribed)
            for t in streaming_tasks:
                t.cancel()
            if streaming_tasks:
                await asyncio.gather(*streaming_tasks, return_exceptions=True)

    def _on_pinch_data_received(self, data_packet: rtc.DataPacket) -> None:
        try:
            raw_bytes = data_packet.data
            try:
                if isinstance(raw_bytes, (bytes, bytearray, memoryview)):
                    raw_str = bytes(raw_bytes).decode("utf-8")
                else:
                    raw_str = str(raw_bytes)
                msg: dict = json.loads(raw_str)
            except Exception as exc:
                logger.warning("Failed to decode Pinch data message: %s", exc)
                return
        except Exception as exc:
            logger.warning("Failed to decode Pinch data message: %s", exc)
            return

        msg_type = msg.get("type", "")
        if msg_type not in ("original_transcript", "translated_transcript"):
            return

        event = TranscriptEvent(
            type=msg_type,
            text=msg.get("text", ""),
            is_final=bool(msg.get("is_final", False)),
            language_detected=msg.get("language_detected", ""),
            timestamp=float(msg.get("timestamp", time.time())),
            confidence=float(msg.get("confidence", 0.0)),
        )

        logger.debug("Transcript [%s] final=%s  %r", event.type, event.is_final, event.text[:80])
        self._emit_transcript(event)

    def _emit_transcript(self, event: TranscriptEvent) -> None:
        for cb in self._transcript_callbacks:
            try:
                cb(event)
            except Exception as exc:
                logger.exception("Unhandled exception in transcript callback %r: %s", cb, exc)

    def __repr__(self) -> str:
        return (
            f"<Translator src={self._options.source_language!r} "
            f"tgt={self._options.target_language!r} "
            f"voice={self._options.voice_type!r} "
            f"started={self._started} stopped={self._stopped}>"
        )
