"""
Blaze TTS Plugin for LiveKit Voice Agent

Text-to-Speech plugin that interfaces with Blaze's TTS service
via the services_gateway WebSocket endpoint.

WebSocket Endpoint: ws://<gateway>/v1/tts/realtime
Protocol:
  1. Connect -> receive {"type": "successful-connection"}
  2. Send {"token": "<auth_token>"} -> receive {"type": "successful-authentication"}
  3. Send {"event": "speech-start", ...params}
  4. Send {"query": "<text>"}
  5. Send {"event": "speech-end"}
  6. Receive JSON {"status": "started-byte-stream"} + binary frames + {"status": "finished-byte-stream"}
Output: Streaming PCM audio chunks
"""

from __future__ import annotations

import asyncio
import json
import re
import time

import httpx
import websockets

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)
from livekit.agents.utils import shortuuid
from livekit.agents.utils.aio.channel import ChanClosed, ChanEmpty

from ._config import BlazeConfig
from ._utils import apply_normalization_rules
from .log import logger

# Regex for splitting text at sentence boundaries.
# Do not split on comma to avoid over-fragmented synthesis and long pauses
# between short clauses under higher-latency environments.
_SENTENCE_END_RE = re.compile(r"(?:\n\n+|\n|[.!?;:。！？；：](?:\s|$))")

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _apply_pcm16_fade(
    pcm_data: bytes,
    *,
    fade_samples: int,
    fade_in: bool = False,
    fade_out: bool = False,
) -> bytes:
    """Apply a short linear fade to 16-bit mono PCM to reduce edge pops."""
    if not pcm_data or (not fade_in and not fade_out):
        return pcm_data

    sample_count = len(pcm_data) // 2
    if sample_count <= 0:
        return pcm_data

    usable_fade = min(fade_samples, sample_count)
    if usable_fade <= 0:
        return pcm_data

    samples = memoryview(bytearray(pcm_data)).cast("h")

    if fade_in:
        for i in range(usable_fade):
            samples[i] = int(samples[i] * (i / usable_fade))

    if fade_out:
        for i in range(usable_fade):
            idx = sample_count - usable_fade + i
            samples[idx] = int(samples[idx] * ((usable_fade - i) / usable_fade))

    return bytes(samples)


def _generate_silence(sample_rate: int, duration_ms: int) -> bytes:
    """Generate silent PCM16 mono audio of the given duration."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * num_samples


# ---------------------------------------------------------------------------
# Batching helpers (used inside SynthesizeStream to combine small text chunks)
# ---------------------------------------------------------------------------


def _find_batch_split(
    text: str,
    *,
    min_chars: int = 100,
    target_chars: int = 200,
    max_chars: int = 350,
    force: bool = False,
    is_first_batch: bool = False,
) -> int | None:
    """Find a natural split point in *text* for TTS batching.

    Returns the character index to split at, or None if the buffer isn't ready yet.

    For the first batch, uses a low word-count threshold to minimise
    first-audio latency without cutting too early by character length.
    Subsequent batches use *min_chars* so that
    short sentences are merged and per-batch WS overhead is reduced.
    """

    def _word_count(s: str) -> int:
        return len(re.findall(r"\S+", s))

    def _safe_split_on_whitespace(s: str, preferred_idx: int, floor_idx: int = 1) -> int:
        """Try to split at a nearby whitespace boundary, otherwise keep preferred index."""
        idx = min(max(preferred_idx, 1), len(s))
        floor = max(1, min(floor_idx, idx))

        # Walk backward to find a whitespace boundary so we don't cut mid-word.
        while idx > floor and not s[idx - 1].isspace():
            idx -= 1

        if idx <= floor:
            return preferred_idx

        # Trim leading spaces from the next chunk.
        while idx < len(s) and s[idx].isspace():
            idx += 1

        return idx

    if not text.strip():
        return None

    hard_limit = min(len(text), max_chars)
    punct_positions = [m.end() for m in _SENTENCE_END_RE.finditer(text[:hard_limit])]

    # First batch: prioritize first-audio latency, but gate by word count
    # instead of fixed character count.
    if is_first_batch:
        for pos in punct_positions:
            if _word_count(text[:pos]) >= 4:
                return pos

    # Hard limit reached: we must split to avoid unbounded buffering.
    if len(text) >= max_chars:
        if punct_positions:
            return punct_positions[-1]
        return _safe_split_on_whitespace(text, max_chars, floor_idx=min_chars)

    # Subsequent batches: prefer a boundary around target size (instead of
    # the earliest >= min_chars) to reduce segment count and WS overhead.
    if len(text) >= min_chars and punct_positions:
        if len(text) >= target_chars:
            for pos in punct_positions:
                if pos >= target_chars:
                    return pos

        candidates = [pos for pos in punct_positions if pos >= min_chars]
        if candidates:
            return candidates[-1]

    # End-of-input flush: always emit remaining content, even if short.
    if force:
        if punct_positions:
            return punct_positions[-1]
        return _safe_split_on_whitespace(text, len(text), floor_idx=1)

    return None


def _normalize_batch_text(text: str) -> str:
    """Final guard before sending text to TTS backend.

    Keeps sentence content intact while removing excessive whitespace/newlines
    that can slow synthesis and produce unnatural pauses.
    """
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


# ---------------------------------------------------------------------------
# TTS Plugin
# ---------------------------------------------------------------------------


class TTS(tts.TTS):
    """
    Blaze Text-to-Speech Plugin with streaming support.

    Uses a single WebSocket connection per turn to synthesize multiple
    text batches, reducing per-sentence connection overhead and enabling
    audio to stream to the user as soon as it's generated.
    """

    def __init__(
        self,
        *,
        api_url: str | None = None,
        language: str = "vi",
        speaker_id: str = "default",
        auth_token: str | None = None,
        model: str = "v1_5_pro",
        audio_format: str = "pcm",
        audio_speed: str = "1",
        audio_quality: int = 32,
        sample_rate: int = 24000,
        normalization_rules: dict[str, str] | None = None,
        batch_min_chars: int = 100,
        batch_target_chars: int = 200,
        batch_max_chars: int = 350,
        batch_max_wait_s: float = 0.45,
        inter_sentence_silence_ms: int = 150,
        timeout: float | None = None,
        config: BlazeConfig | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._config = config or BlazeConfig()

        self._api_url = api_url or self._config.api_url
        self._language = language
        self._speaker_id = speaker_id
        self._auth_token = auth_token or self._config.api_token
        self._model = model
        normalized_format = (audio_format or "pcm").strip().lower()
        if normalized_format not in {"pcm", "mp3", "wav"}:
            logger.warning("Unsupported audio_format '%s', falling back to 'pcm'", audio_format)
            normalized_format = "pcm"
        self._audio_format = normalized_format
        self._audio_speed = audio_speed
        self._audio_quality = int(audio_quality) if audio_quality is not None else 32
        self._sample_rate = sample_rate
        self._timeout = timeout if timeout is not None else self._config.tts_timeout
        self._normalization_rules = normalization_rules

        # Batching configuration
        self._batch_min_chars = batch_min_chars
        self._batch_target_chars = batch_target_chars
        self._batch_max_chars = batch_max_chars
        self._batch_max_wait_s = batch_max_wait_s
        self._inter_sentence_silence_ms = inter_sentence_silence_ms

        # Build WebSocket URL (convert http(s) to ws(s))
        ws_base = self._api_url.replace("https://", "wss://").replace("http://", "ws://")
        self._ws_url = f"{ws_base}/v1/tts/realtime"

        logger.info(
            f"BlazeTTS initialized (streaming): url={self._api_url}, "
            f"speaker={self._speaker_id}, language={self._language}, format={self._audio_format}, "
            f"batch=[{batch_min_chars},{batch_target_chars},{batch_max_chars}]"
        )

    @property
    def provider(self) -> str:
        return "Blaze"

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await super().aclose()

    def update_options(
        self,
        *,
        speaker_id: str | None = None,
        model: str | None = None,
        audio_format: str | None = None,
        audio_quality: int | None = None,
        language: str | None = None,
        auth_token: str | None = None,
        normalization_rules: dict[str, str] | None = None,
    ) -> None:
        if speaker_id is not None:
            self._speaker_id = speaker_id
        if model is not None:
            self._model = model
        if audio_format is not None:
            fmt = (audio_format or "pcm").strip().lower()
            if fmt not in {"pcm", "mp3", "wav"}:
                logger.warning(
                    "Unsupported audio_format '%s', keeping current '%s'",
                    audio_format,
                    self._audio_format,
                )
            else:
                self._audio_format = fmt
        if audio_quality is not None:
            self._audio_quality = int(audio_quality)
        if language is not None:
            self._language = language
        if auth_token is not None:
            self._auth_token = auth_token
        if normalization_rules is not None:
            self._normalization_rules = normalization_rules

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """Non-streaming synthesis via HTTP POST.

        Uses a direct HTTP request instead of WebSocket, suitable for
        short texts or environments where WebSocket is not available.
        """
        return ChunkedStream(
            tts_instance=self,
            input_text=text,
            conn_options=conn_options,
        )

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _TTSSynthesizeStream:
        """Create a streaming TTS session backed by a single WebSocket."""
        return _TTSSynthesizeStream(tts_instance=self, conn_options=conn_options)


# ---------------------------------------------------------------------------
# ChunkedStream — one-shot HTTP POST synthesis
# ---------------------------------------------------------------------------


class ChunkedStream(tts.ChunkedStream):
    """Non-streaming TTS via HTTP POST.

    Sends the full text in a single request and streams back audio chunks.
    Useful as a fallback when WebSocket is unavailable or for short texts.

    API Endpoint: POST /v1/tts/realtime (multipart form)
    """

    def __init__(
        self,
        tts_instance: TTS,
        *,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts_instance, input_text=input_text, conn_options=conn_options)
        self._blaze_tts = tts_instance

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = shortuuid()
        tts_cfg = self._blaze_tts

        synthesize_url = f"{tts_cfg._api_url}/v1/tts/realtime"
        text = apply_normalization_rules(self._input_text, tts_cfg._normalization_rules)
        if not text.strip():
            return

        headers: dict[str, str] = {}
        if tts_cfg._auth_token:
            headers["Authorization"] = f"Bearer {tts_cfg._auth_token}"

        form_data = {
            "query": text,
            "language": tts_cfg._language,
            "audio_format": tts_cfg._audio_format,
            "speaker_id": tts_cfg._speaker_id,
            "model": tts_cfg._model,
        }
        if tts_cfg._audio_speed and tts_cfg._audio_speed != "1":
            form_data["audio_speed"] = tts_cfg._audio_speed
        if tts_cfg._audio_quality is not None:
            form_data["audio_quality"] = str(tts_cfg._audio_quality)

        mime_type = {
            "pcm": "audio/pcm",
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
        }.get(tts_cfg._audio_format, "audio/pcm")

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=tts_cfg._sample_rate,
            num_channels=1,
            mime_type=mime_type,
            stream=False,
        )
        segment_id = shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        logger.info(
            "[%s] TTS chunked request: %d chars, speaker=%s",
            request_id,
            len(text),
            tts_cfg._speaker_id,
        )

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(tts_cfg._timeout, connect=5.0)
            ) as client:
                async with client.stream(
                    "POST",
                    synthesize_url,
                    data=form_data,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        error_text = (await response.aread()).decode(errors="replace")
                        raise APIStatusError(
                            f"TTS service error {response.status_code}: {error_text}",
                            status_code=response.status_code,
                            request_id=request_id,
                            body=error_text,
                        )

                    async for chunk in response.aiter_bytes(4096):
                        if chunk:
                            output_emitter.push(chunk)

        except httpx.TimeoutException as e:
            raise APITimeoutError(f"TTS request timed out: {e}") from e
        except httpx.NetworkError as e:
            raise APIConnectionError(f"TTS network error: {e}") from e
        except (APIStatusError, APITimeoutError, APIConnectionError):
            raise
        except Exception as e:
            raise APIConnectionError(f"TTS connection error: {e}") from e

        output_emitter.flush()
        logger.info("[%s] TTS chunked synthesis complete", request_id)


# ---------------------------------------------------------------------------
# Streaming TTS — single WebSocket, multiple batches
# ---------------------------------------------------------------------------


class _TTSSynthesizeStream(tts.SynthesizeStream):
    """Streaming TTS that reuses one WebSocket for all batches in a turn."""

    def __init__(
        self,
        tts_instance: TTS,
        *,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts_instance, conn_options=conn_options)
        self._blaze_tts = tts_instance

    # ------------------------------------------------------------------
    # WebSocket helpers
    # ------------------------------------------------------------------

    def _speech_start_params(self) -> dict:
        """Build the speech-start message payload."""
        params: dict = {
            "event": "speech-start",
            "language": self._blaze_tts._language,
            "audio_format": self._blaze_tts._audio_format,
            "speaker_id": self._blaze_tts._speaker_id,
            "normalization": "no",
            "model": self._blaze_tts._model,
        }
        if self._blaze_tts._audio_speed and self._blaze_tts._audio_speed != "1":
            params["audio_speed"] = self._blaze_tts._audio_speed
        if self._blaze_tts._audio_quality is not None:
            params["audio_quality"] = self._blaze_tts._audio_quality
        return params

    async def _ws_connect_and_auth(self, ws: websockets.ClientConnection, request_id: str) -> None:
        """Perform the initial WS handshake: connection ack + authentication."""
        auth_start = time.monotonic()

        msg = json.loads(await ws.recv())
        if msg.get("type") != "successful-connection":
            raise APIConnectionError(f"Unexpected WS message on connect: {msg}")
        logger.debug("[%s] TTS WS connected: %s", request_id, msg)

        await ws.send(
            json.dumps(
                {
                    "token": self._blaze_tts._auth_token or "",
                    "strategy": "livekit",
                }
            )
        )
        msg = json.loads(await ws.recv())
        if msg.get("type") != "successful-authentication":
            raise APIStatusError(
                f"WS authentication failed: {msg}",
                status_code=403,
                request_id=request_id,
                body=json.dumps(msg),
            )

        logger.info(
            "[%s] TTS WS connected & authenticated in %.3fs — url=%s, speaker=%s, model=%s",
            request_id,
            time.monotonic() - auth_start,
            self._blaze_tts._ws_url,
            self._blaze_tts._speaker_id,
            self._blaze_tts._model,
        )

    # ------------------------------------------------------------------
    # Main streaming loop — single speech session per turn
    # ------------------------------------------------------------------

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Read text tokens, batch them, synthesize over a single WS speech session.

        Sends one speech-start / speech-end pair for the entire turn so the
        gateway only extracts 'first N words' once, reducing external TTS API
        calls from ~2×batches to ~batches+1.  Audio reading runs concurrently
        with text sending for optimal pipelining.
        """
        request_id = shortuuid()
        turn_start = time.monotonic()
        tts_cfg = self._blaze_tts
        configured_mime_type = {
            "pcm": "audio/pcm",
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
        }.get(tts_cfg._audio_format, "audio/pcm")
        stream_initialized = False
        runtime_mime_type = configured_mime_type
        runtime_is_pcm = runtime_mime_type == "audio/pcm"

        def _normalize_mime_from_content_type(content_type: str | None) -> str:
            if not content_type:
                return configured_mime_type
            lowered = content_type.lower()
            if "audio/mpeg" in lowered or "audio/mp3" in lowered:
                return "audio/mpeg"
            if "audio/wav" in lowered or "audio/x-wav" in lowered:
                return "audio/wav"
            if "audio/pcm" in lowered or "audio/l16" in lowered:
                return "audio/pcm"
            return configured_mime_type

        def _ensure_output_initialized(
            detected_content_type: str | None = None,
        ) -> None:
            nonlocal stream_initialized, runtime_mime_type, runtime_is_pcm
            if stream_initialized:
                return

            runtime_mime_type = _normalize_mime_from_content_type(detected_content_type)
            runtime_is_pcm = runtime_mime_type == "audio/pcm"

            if runtime_mime_type != configured_mime_type:
                logger.warning(
                    "[%s] TTS mime mismatch: configured=%s, detected=%s. Using detected mime.",
                    request_id,
                    configured_mime_type,
                    runtime_mime_type,
                )

            output_emitter.initialize(
                request_id=request_id,
                sample_rate=tts_cfg._sample_rate,
                num_channels=1,
                mime_type=runtime_mime_type,
                stream=True,
            )

            segment_id = shortuuid()
            output_emitter.start_segment(segment_id=segment_id)
            stream_initialized = True

        batch_count = 0
        seg_count = 0

        try:
            async with asyncio.timeout(tts_cfg._timeout):
                async with websockets.connect(tts_cfg._ws_url) as ws:
                    await self._ws_connect_and_auth(ws, request_id)

                    # ---------- single speech-start for the whole turn ----------
                    speech_start_msg = self._speech_start_params()
                    logger.info(
                        "[%s] TTS speech-start (single session): %s",
                        request_id,
                        json.dumps(speech_start_msg, ensure_ascii=False),
                    )
                    await ws.send(json.dumps(speech_start_msg))
                    ack = await ws.recv()
                    logger.debug("[%s] TTS speech-start ack: %s", request_id, ack)

                    # ---------- concurrent audio reader ----------
                    async def _read_audio() -> tuple[int, int]:
                        """Read all WS responses until speech-end.

                        Returns (tts_segment_count, total_audio_bytes).
                        """
                        _seg_count = 0
                        total_bytes = 0
                        pending_tail = b""
                        first_audio = True
                        first_audio_t: float | None = None
                        has_prev_segment = False
                        fade_samples = max(1, int(tts_cfg._sample_rate * 0.008))
                        tail_bytes = fade_samples * 2
                        silence_pcm = _generate_silence(
                            tts_cfg._sample_rate,
                            tts_cfg._inter_sentence_silence_ms,
                        )

                        while True:
                            frame = await ws.recv()

                            if isinstance(frame, bytes):
                                if not frame:
                                    continue
                                _ensure_output_initialized()
                                if first_audio_t is None:
                                    first_audio_t = time.monotonic()
                                    logger.info(
                                        "[%s] TTS first audio: %.3fs after turn start",
                                        request_id,
                                        first_audio_t - turn_start,
                                    )
                                total_bytes += len(frame)

                                if not runtime_is_pcm:
                                    output_emitter.push(frame)
                                    first_audio = False
                                    continue

                                pcm = pending_tail + frame
                                if first_audio:
                                    pcm = _apply_pcm16_fade(
                                        pcm,
                                        fade_samples=fade_samples,
                                        fade_in=True,
                                    )
                                first_audio = False

                                if len(pcm) <= tail_bytes:
                                    pending_tail = pcm
                                    continue
                                output_emitter.push(pcm[:-tail_bytes])
                                pending_tail = pcm[-tail_bytes:]
                            else:
                                msg = json.loads(frame)
                                st = msg.get("status") or msg.get("type", "")

                                if st == "started-byte-stream":
                                    _ensure_output_initialized(msg.get("contentType"))

                                if st == "speech-end":
                                    # Final flush with fade-out
                                    if runtime_is_pcm and pending_tail:
                                        output_emitter.push(
                                            _apply_pcm16_fade(
                                                pending_tail,
                                                fade_samples=fade_samples,
                                                fade_out=True,
                                            )
                                        )
                                    break

                                if st == "started-byte-stream":
                                    # Silence between TTS segments (not before the first)
                                    if runtime_is_pcm and has_prev_segment and silence_pcm:
                                        output_emitter.push(silence_pcm)

                                elif st == "finished-byte-stream":
                                    _seg_count += 1
                                    # Fade-out tail of this TTS segment
                                    if runtime_is_pcm and pending_tail:
                                        output_emitter.push(
                                            _apply_pcm16_fade(
                                                pending_tail,
                                                fade_samples=fade_samples,
                                                fade_out=True,
                                            )
                                        )
                                        pending_tail = b""
                                    has_prev_segment = True
                                    logger.debug(
                                        "[%s] TTS segment %d done",
                                        request_id,
                                        _seg_count,
                                    )

                                elif st in ("failed-request", "error"):
                                    raise APIStatusError(
                                        f"TTS failed: {msg.get('message', '')} "
                                        f"{msg.get('details', '')}",
                                        status_code=500,
                                        request_id=request_id,
                                        body=json.dumps(msg),
                                    )
                                # processing-request, speech-start → skip

                        return _seg_count, total_bytes

                    reader_task = asyncio.create_task(_read_audio())

                    # ---------- batch text sender ----------
                    async def _send_query(text: str) -> None:
                        nonlocal batch_count
                        normalized = apply_normalization_rules(
                            text,
                            tts_cfg._normalization_rules,
                        )
                        normalized = _normalize_batch_text(normalized)
                        if not normalized.strip():
                            return
                        batch_count += 1
                        has_img_tag = (
                            "<img>" in normalized.lower() or "</img>" in normalized.lower()
                        )
                        preview = normalized[:80] + ("..." if len(normalized) > 80 else "")
                        logger.info(
                            "[%s] TTS batch %d — %d chars has_img_tag=%s: '%s'",
                            request_id,
                            batch_count,
                            len(normalized),
                            has_img_tag,
                            preview,
                        )
                        logger.debug(
                            "[%s] TTS batch %d full_text=%r",
                            request_id,
                            batch_count,
                            normalized,
                        )
                        await ws.send(json.dumps({"query": normalized}))

                    text_buf = ""
                    input_done = False

                    async def _drain_batches(force: bool) -> None:
                        nonlocal text_buf
                        while True:
                            idx = _find_batch_split(
                                text_buf,
                                min_chars=tts_cfg._batch_min_chars,
                                target_chars=tts_cfg._batch_target_chars,
                                max_chars=tts_cfg._batch_max_chars,
                                force=force,
                                is_first_batch=(batch_count == 0),
                            )
                            if idx is None:
                                break
                            chunk = text_buf[:idx]
                            text_buf = text_buf[idx:]
                            if not chunk:
                                continue
                            if len(chunk.strip()) < 8 and not force:
                                text_buf = chunk + text_buf
                                break
                            await _send_query(chunk)

                    try:
                        # Main input loop — read tokens with a timeout so we
                        # can flush accumulated text while waiting for more.
                        while not input_done:
                            try:
                                data = await asyncio.wait_for(
                                    self._input_ch.recv(),
                                    timeout=tts_cfg._batch_max_wait_s,
                                )
                            except asyncio.TimeoutError:
                                await _drain_batches(force=False)
                                continue
                            except (ChanClosed, StopAsyncIteration):
                                input_done = True
                                await _drain_batches(force=True)
                                break

                            if isinstance(data, self._FlushSentinel):
                                input_done = True
                                await _drain_batches(force=True)
                                break

                            text_buf += data

                            # Read-ahead: drain pending tokens so short
                            # sentences get merged before we split again.
                            while True:
                                try:
                                    extra = self._input_ch.recv_nowait()
                                except (ChanEmpty, ChanClosed):
                                    break
                                if isinstance(extra, self._FlushSentinel):
                                    input_done = True
                                    break
                                text_buf += extra

                            await _drain_batches(force=input_done)

                        # Flush any remaining text
                        if text_buf:
                            await _send_query(text_buf)

                        # Close the single speech session
                        await ws.send(json.dumps({"event": "speech-end"}))

                    except Exception:
                        reader_task.cancel()
                        raise

                    # Wait for all audio to arrive
                    seg_count, _ = await reader_task

                    if not stream_initialized:
                        # No audio arrived, but keep emitter lifecycle consistent.
                        _ensure_output_initialized()

        except TimeoutError as e:
            raise APITimeoutError(f"TTS stream timed out: {e}") from e
        except websockets.exceptions.ConnectionClosed as e:
            raise APIConnectionError(f"TTS WebSocket closed: {e}") from e
        except (APIStatusError, APITimeoutError, APIConnectionError):
            raise
        except Exception as e:
            raise APIConnectionError(f"TTS stream error: {e}") from e

        output_emitter.flush()

        logger.info(
            "[%s] TTS turn complete: %d batches, %d TTS segments, %.3fs total",
            request_id,
            batch_count,
            seg_count,
            time.monotonic() - turn_start,
        )
