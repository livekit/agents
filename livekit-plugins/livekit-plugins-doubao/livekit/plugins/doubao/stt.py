            import gzip
          from livekit.agents.vad import silero
from .log import logger
from __future__ import annotations
from dataclasses import dataclass
from livekit import rtc
from livekit.agents import (
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, http_context, is_given
from typing import Any
import aiohttp
import asyncio
import json
import struct
import time
import uuid

    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
)
# Defaults are intentionally conservative and overridable via env/args
_DEFAULT_STT_ENDPOINT = os.environ.get(
    "DOUBAO_STT_ENDPOINT",
    # SAUC optimized bidirectional streaming endpoint per official docs
    "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
)

def _env_int(name: str) -> int | None:
    v = os.environ.get(name)
    try:
        return int(v) if v is not None and str(v).strip() != "" else None
    except Exception:
        return None


def _env_float(name: str, default: float | None = None) -> float | None:
    v = os.environ.get(name)
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def _env_bool(name: str) -> bool | None:
    v = os.environ.get(name)
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):  # truthy
        return True
    if s in ("0", "false", "no", "of"):  # falsy
        return False
    return None


# Centralized environment-driven tuning knobs
ENV_RESULT_TYPE = os.environ.get("DOUBAO_STT_RESULT_TYPE", "full")
ENV_END_WINDOW_MS = _env_int("DOUBAO_STT_END_WINDOW_MS") or 800
ENV_FORCE_TO_SPEECH_MS = _env_int("DOUBAO_STT_FORCE_TO_SPEECH_MS") or 1000
ENV_VAD_SEGMENT_MS = _env_int("DOUBAO_STT_VAD_SEGMENT_MS") or 3000
ENV_ENABLE_NONSTREAM = _env_bool("DOUBAO_STT_ENABLE_NONSTREAM")  # tri-state: None -> use default
ENV_ENABLE_ACCELERATE = _env_bool("DOUBAO_STT_ENABLE_ACCELERATE")
ENV_ACCELERATE_SCORE = _env_int("DOUBAO_STT_ACCELERATE_SCORE")

ENV_SEG_MS = _env_int("DOUBAO_STT_SEG_MS") or 200
ENV_SINGLE_PACKET = (_env_bool("DOUBAO_STT_SINGLE_PACKET") is True)
ENV_USAGE_PERIOD = _env_float("DOUBAO_STT_USAGE_PERIOD", 5.0) or 5.0

# Connection keepalive and timeout settings
ENV_WS_TIMEOUT = _env_int("DOUBAO_STT_WS_TIMEOUT") or 60  # WebSocket timeout in seconds
ENV_MAX_IDLE_MS = _env_int("DOUBAO_STT_MAX_IDLE_MS") or 30000  # Max idle time before sending keepalive

@dataclass
class _STTOptions:
    app_id: str
    access_token: str
    endpoint: str
    resource_id: str | None
    language_code: str
    sample_rate: int
    encoding: str  # wav | pcm | mp3 (input encoding; we use wav)


class STT(stt.STT):
    """Doubao/Volc OpenSpeech STT (short-audio over WebSocket)

    This mirrors the LiveKit STT interface and uses a unidirectional WS flow,
    modeled after the TTS implementation and the official SAUC example.

    Configure via either constructor args or env vars:
      - DOUBAO_APP_ID
      - DOUBAO_ACCESS_TOKEN (or DOUBAO_APP_ACCESS_TOKEN)
      - DOUBAO_STT_ENDPOINT (optional)
      - DOUBAO_STT_LANGUAGE (optional, default: zh)
      - DOUBAO_STT_RESOURCE_ID (optional)
    """

    def __init__(
        self,
        *,
        app_id: NotGivenOr[str] = NOT_GIVEN,
        access_token: NotGivenOr[str] = NOT_GIVEN,
        endpoint: NotGivenOr[str] = NOT_GIVEN,
        resource_id: NotGivenOr[str] = NOT_GIVEN,
        language_code: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        encoding: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        # Advertise streaming support; we'll provide a RecognizeStream implementation
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=True))

        app_id_val = app_id if is_given(app_id) else os.environ.get("DOUBAO_APP_ID")
        access_token_val = (
            access_token
            if is_given(access_token)
            else (
                os.environ.get("DOUBAO_ACCESS_TOKEN")
                or os.environ.get("DOUBAO_APP_ACCESS_TOKEN")
            )
        )
        if not app_id_val or not access_token_val:
            raise ValueError(
                "Doubao STT requires app_id and access_token (DOUBAO_APP_ID/DOUBAO_ACCESS_TOKEN)"
            )

        self._opts = _STTOptions(
            app_id=app_id_val,
            access_token=access_token_val,
            endpoint=(endpoint if is_given(endpoint) else _DEFAULT_STT_ENDPOINT) or _DEFAULT_STT_ENDPOINT,
            resource_id=(resource_id if is_given(resource_id) else os.environ.get("DOUBAO_STT_RESOURCE_ID")),
            language_code=(language_code if is_given(language_code) else os.environ.get("DOUBAO_STT_LANGUAGE", "zh")),
            sample_rate=int(sample_rate) if is_given(sample_rate) else 16000,
            encoding=(encoding if is_given(encoding) else "wav"),
        )

        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        if is_given(language):
            self._opts.language_code = language

        # Prepare short-audio wav bytes
        # Ensure sample rate matches SAUC config (default 16k)
        combined = rtc.combine_audio_frames(buffer)
        if combined.sample_rate != self._opts.sample_rate:
            resampler = rtc.AudioResampler(
                combined.sample_rate, self._opts.sample_rate, quality=rtc.AudioResamplerQuality.HIGH
            )
            out_frames = resampler.push(combined)
            for extra in resampler.flush():
                out_frames.append(extra)
            if not out_frames:
                raise APIConnectionError("resampling produced no frames")
            combined = rtc.combine_audio_frames(out_frames)
        wav_bytes = combined.to_wav_bytes()

        # SAUC auth headers per demo
        headers = {
            "X-Api-Resource-Id": self._opts.resource_id or os.environ.get("DOUBAO_STT_RESOURCE_ID", "volc.bigasr.sauc.duration"),
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Access-Key": self._opts.access_token,
            "X-Api-App-Key": self._opts.app_id,
        }

        # Helper builders (inline from demo)
        PROTOCOL_VERSION = 0b0001
        MT_CLIENT_FULL = 0b0001
        MT_CLIENT_AUDIO = 0b0010
        MTS_POS_SEQ = 0b0001
        MTS_NEG_WITH_SEQ = 0b0011
        SER_JSON = 0b0001
        SER_NONE = 0b0000
        CMP_GZIP = 0b0001

        def header_bytes(message_type: int, flags: int, serialization: int, compression: int) -> bytes:
            b = bytearray()
            b.append((PROTOCOL_VERSION << 4) | 1)
            b.append((message_type << 4) | flags)
            b.append((serialization << 4) | compression)
            b.append(0)
            return bytes(b)

        def full_client_request(seq: int, *, nostream: bool) -> bytes:
            hdr = header_bytes(MT_CLIENT_FULL, MTS_POS_SEQ, SER_JSON, CMP_GZIP)
            # For nostream endpoint, do not enable_nonstream; for streaming endpoints it is ignored here
            payload = {
                "user": {"uid": "demo_uid"},
                "audio": {
                    "format": "wav",
                    "codec": "raw",
                    "rate": self._opts.sample_rate,
                    "bits": 16,
                    "channel": 1,
                },
                "request": {
                    "model_name": "bigmodel",
                    "enable_itn": True,
                    "enable_punc": True,
                    "enable_ddc": True,
                    "show_utterances": True,
                    "enable_nonstream": ENV_ENABLE_NONSTREAM,
                    "result_type": "full",
                    "enable_emotion_detection": True,
                    "enable_gender_detection": True
                },
            }
            # language is supported only for nostream per docs
            if nostream and self._opts.language_code:
                payload["audio"]["language"] = self._opts.language_code
            comp = __import__("gzip").compress(json.dumps(payload).encode("utf-8"))
            return b"".join([hdr, struct.pack(">i", seq), struct.pack(">I", len(comp)), comp])

        def audio_only_request(seq: int, chunk: bytes, is_last: bool) -> bytes:
            # Use last-without-seq per docs when finalizing
            MTS_LAST_NO_SEQ = 0b0010
            flags = MTS_LAST_NO_SEQ if is_last else MTS_POS_SEQ
            hdr = header_bytes(MT_CLIENT_AUDIO, flags, SER_NONE, CMP_GZIP)
            comp = __import__("gzip").compress(chunk)
            if is_last:
                return b"".join([hdr, struct.pack(">I", len(comp)), comp])
            else:
                return b"".join([hdr, struct.pack(">i", seq), struct.pack(">I", len(comp)), comp])

        # SAUC demo sends full WAV bytes (including header). Follow the same.
        payload_bytes = wav_bytes

        # Choose nostream endpoint for prerecorded recognition per docs
        ws_url = self._opts.endpoint
        if "bigmodel_nostream" not in (ws_url or ""):
            ws_url = ws_url.replace("bigmodel_async", "bigmodel_nostream").replace("bigmodel", "bigmodel_nostream")

        # Connect WS
        session = self._ensure_session()
        try:
            ws = await session.ws_connect(
                ws_url,
                headers=headers,
                timeout=conn_options.timeout,
                max_msg_size=10 * 1024 * 1024,
            )
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            raise APIConnectionError("could not connect to Doubao STT") from e

        # capture request id if provided (used by metrics)
        request_id = ""
        try:
            request_id = getattr(ws, "_response").headers.get("X-Tt-Logid", "")
        except Exception:
            request_id = ""

        final_text: str | None = None
        start_time = 0.0
        end_time = 0.0
        speaker_id: str | None = None

        try:
            # Send config first and wait for the initial server response, as per official demo
            await ws.send_bytes(full_client_request(1, nostream=True))

            # Expect a binary ack before streaming audio; if an error frame is received, raise with details
            first = await ws.receive()
            if first.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                code = ws.close_code
                reason = ws.close_message
                raise APIStatusError(f"connection closed during handshake (code={code}, reason={reason})")
            if first.type == aiohttp.WSMsgType.BINARY:
                fdata = first.data
                f_header_size_words = fdata[0] & 0x0F
                f_message_type = fdata[1] >> 4
                f_flags = fdata[1] & 0x0F
                f_compression = fdata[2] & 0x0F
                f_payload = fdata[f_header_size_words * 4 :]
                # strip seq/event if present
                if f_flags & 0x01:
                    f_payload = f_payload[4:]
                if f_flags & 0x04:
                    # event code
                    f_payload = f_payload[4:]
                # full response has only size then payload; error response has code then size
                try:
                    if f_message_type == 0xF:  # SERVER_ERROR_RESPONSE
                        err_code = struct.unpack(">i", f_payload[:4])[0]
                        payload_size = struct.unpack(">I", f_payload[4:8])[0]
                        f_payload = f_payload[8:]
                    else:
                        payload_size = struct.unpack(">I", f_payload[:4])[0]
                        f_payload = f_payload[4:]
                        err_code = 0
                except Exception:
                    err_code = -1
                    payload_size = 0
                if f_compression == 0x1:
                    try:
                        f_payload = __import__("gzip").decompress(f_payload)
                    except Exception:
                        pass
                # If error frame, surface error details early
                if 'err_code' in locals() and err_code and err_code != 0:
                    err_msg = None
                    try:
                        jobj = json.loads(f_payload.decode("utf-8", errors="ignore")) if payload_size else {}
                        err_msg = jobj.get("message") or jobj.get("msg") or jobj.get("error")
                    except Exception:
                        err_msg = None
                    raise APIStatusError(
                        message=f"handshake error: code={err_code}, msg={err_msg}",
                        status_code=err_code,
                        request_id=None,
                        body=None,
                    )

            # Concurrently send audio segments while receiving responses
            seg_ms = int(ENV_SEG_MS)
            bytes_per_sec = self._opts.sample_rate * 1 * 2
            seg_bytes = max(1, (bytes_per_sec * seg_ms) // 1000)
            single_packet = ENV_SINGLE_PACKET

            async def _sender() -> None:
                total = len(payload_bytes)
                pos = 0
                # After sending the full client request with seq=1, audio packets must start at seq=2
                seq = 2
                try:
                    if single_packet:
                        await ws.send_bytes(audio_only_request(seq, payload_bytes, True))
                    else:
                        while pos < total and not ws.closed:
                            end = min(total, pos + seg_bytes)
                            is_last = end >= total
                            await ws.send_bytes(
                                audio_only_request(seq, payload_bytes[pos:end], is_last)
                            )
                            if not is_last:
                                seq += 1
                            pos = end
                            # match demo pacing to be safe
                            await asyncio.sleep(seg_ms / 1000)
                except Exception:
                    # ignore send errors if ws is closing
                    pass

            send_task = asyncio.create_task(_sender())

            # Receive loop
            while True:
                incoming = await ws.receive()
                if incoming.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    code = ws.close_code
                    reason = ws.close_message
                    raise APIStatusError(f"connection closed (code={code}, reason={reason})")
                if incoming.type != aiohttp.WSMsgType.BINARY:
                    continue

                data = incoming.data
                header_size_words = data[0] & 0x0F
                message_type = data[1] >> 4
                flags = data[1] & 0x0F
                serialization = data[2] >> 4
                compression = data[2] & 0x0F
                payload = data[header_size_words * 4 :]

                if flags & 0x01:
                    # sequence present
                    payload = payload[4:]
                if flags & 0x02:
                    # last packet flag; we'll use it to stop after parsing
                    is_last = True
                else:
                    is_last = False
                if flags & 0x04:
                    # event code (int32) present
                    payload = payload[4:]

                size = 0
                # For error frames, there is an int32 error code before the size
                err_code: int | None = None
                if message_type == 0xF and len(payload) >= 8:
                    err_code = struct.unpack(">i", payload[:4])[0]
                    size = struct.unpack(">I", payload[4:8])[0]
                    payload = payload[8:]
                elif len(payload) >= 4:
                    size = struct.unpack(">I", payload[:4])[0]
                    payload = payload[4:]

                if compression == 0x1:  # gzip
                    try:
                        payload = __import__("gzip").decompress(payload)
                    except Exception:
                        pass

                obj: dict[str, Any] = {}
                if serialization == 0x1 and size:
                    try:
                        obj = json.loads(payload.decode("utf-8", errors="ignore"))
                    except Exception:
                        obj = {}

                # If server reported error, surface it with details
                if err_code is not None and err_code != 0:
                    msg = obj.get("message") if isinstance(obj, dict) else None
                    raise APIStatusError(
                        message=f"server error: code={err_code}, msg={msg}",
                        status_code=err_code,
                        request_id=None,
                        body=obj,
                    )

                if obj:
                    # Try to extract final text from multiple possible shapes
                    txt: str | None = None
                    # direct string fields
                    for k in ("text", "transcript", "caption"):
                        v = obj.get(k)
                        if isinstance(v, str) and v.strip():
                            txt = v
                            break

                    # nested result forms
                    if not txt:
                        res = obj.get("result")
                        if isinstance(res, str) and res.strip():
                            txt = res
                        elif isinstance(res, dict):
                            for kk in ("text", "sentence", "transcript", "best_result"):
                                vv = res.get(kk)
                                if isinstance(vv, str) and vv.strip():
                                    txt = vv
                                    break
                        elif isinstance(res, list) and res:
                            # list of segments
                            parts: list[str] = []
                            for it in res:
                                if isinstance(it, dict):
                                    for kk in ("text", "sentence", "transcript"):
                                        vv = it.get(kk)
                                        if isinstance(vv, str) and vv.strip():
                                            parts.append(vv)
                                            break
                            if parts:
                                txt = "".join(parts)

                    # best_result variants
                    if not txt:
                        br = obj.get("best_result")
                        if isinstance(br, str) and br.strip():
                            txt = br
                        elif isinstance(br, dict):
                            for kk in ("sentence", "text", "transcript"):
                                vv = br.get(kk)
                                if isinstance(vv, str) and vv.strip():
                                    txt = vv
                                    break

                    # nbest candidates
                    if not txt and isinstance(obj.get("nbest"), list) and obj["nbest"]:
                        cand = obj["nbest"][0]
                        if isinstance(cand, dict):
                            for kk in ("sentence", "text", "transcript"):
                                vv = cand.get(kk)
                                if isinstance(vv, str) and vv.strip():
                                    txt = vv
                                    break

                    # utterances list
                    if not txt and isinstance(obj.get("utterances"), list) and obj["utterances"]:
                        try:
                            texts = [u.get("text", "") for u in obj["utterances"] if isinstance(u, dict)]
                            candidate = "".join(t for t in texts if t)
                            if candidate.strip():
                                txt = candidate
                        except Exception:
                            pass

                    # tokens/words list
                    words = obj.get("words") or obj.get("tokens")
                    if isinstance(words, list) and words:
                        try:
                            # timestamps
                            start_time = min(float(w.get("start", 0)) for w in words)
                            end_time = max(float(w.get("end", 0)) for w in words)
                        except Exception:
                            pass
                        speaker_id = words[0].get("speaker_id") if isinstance(words[0], dict) else None
                        if not txt:
                            try:
                                tokens = [
                                    (w.get("text") or w.get("token") or w.get("word") or "")
                                    for w in words
                                    if isinstance(w, dict)
                                ]
                                candidate = "".join(t for t in tokens if isinstance(t, str))
                                if candidate.strip():
                                    txt = candidate
                            except Exception:
                                pass

                    if isinstance(txt, str) and txt.strip():
                        final_text = txt

                if is_last:
                    break

        finally:
            try:
                await send_task
            except Exception:
                pass
            await ws.close()

        # Fallback if nothing was parsed
        if not isinstance(final_text, str):
            final_text = ""

        return stt.SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            request_id=request_id,
            alternatives=[
                stt.SpeechData(
                    text=final_text,
                    language=self._opts.language_code,
                    speaker_id=speaker_id,
                    start_time=start_time,
                    end_time=end_time,
                )
            ],
        )

    # =============== Streaming (WebSocket) implementation ===============
    class _SAUCStream(stt.RecognizeStream):
        def __init__(self, *, stt: "STT", conn_options: APIConnectOptions, sample_rate: NotGivenOr[int] = NOT_GIVEN) -> None:
            super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
            self._parent: STT = stt
            self._ws: aiohttp.ClientWebSocketResponse | None = None
            self._seq: int = 1
            self._running = True
            self._request_id: str = ""
            self._closing: bool = False
            # Track incremental emission to avoid duplicate text to the AgentSession
            self._emitted_definite_count: int = 0
            self._last_interim_text: str = ""
            self._final_accum_text: str = ""
            # Track speaking state to throttle metrics like other plugins
            self._speaking: bool = False
            self._usage_accum: float = 0.0
            self._usage_period = ENV_USAGE_PERIOD
            self._last_usage_emit = time.monotonic()
            # Connection health tracking
            self._last_activity_time: float = time.monotonic()

        def _gzip(self, data: bytes) -> bytes:
            return gzip.compress(data)

        def _header_bytes(self, message_type: int, flags: int, serialization: int, compression: int) -> bytes:
            PROTOCOL_VERSION = 0b0001
            b = bytearray()
            b.append((PROTOCOL_VERSION << 4) | 1)
            b.append((message_type << 4) | flags)
            b.append((serialization << 4) | compression)
            b.append(0)
            return bytes(b)

        def _full_client_request(self, seq: int) -> bytes:
            MT_CLIENT_FULL = 0b0001
            MTS_POS_SEQ = 0b0001
            SER_JSON = 0b0001
            CMP_GZIP = 0b0001

            # Follow official demo: include positive sequence for the full client request
            hdr = self._header_bytes(MT_CLIENT_FULL, MTS_POS_SEQ, SER_JSON, CMP_GZIP)
            payload = {
                "user": {"uid": "livekit_stt"},
                "audio": {
                    "format": "pcm",
                    "codec": "raw",
                    "rate": self._parent._opts.sample_rate,
                    "bits": 16,
                    "channel": 1,
                },
                "request": {
                    "model_name": "bigmodel",
                    "enable_itn": True,
                    "enable_punc": True,
                    "enable_ddc": True,
                    "show_utterances": True,
                    "result_type": ENV_RESULT_TYPE,
                },
            }
            # default to enabling two-pass recognition on streaming bigmodel_async per docs
            if ENV_ENABLE_NONSTREAM is None:
                payload["request"]["enable_nonstream"] = True
            else:
                payload["request"]["enable_nonstream"] = bool(ENV_ENABLE_NONSTREAM)
            if ENV_END_WINDOW_MS is not None:
                payload["request"]["end_window_size"] = int(ENV_END_WINDOW_MS)
            if ENV_FORCE_TO_SPEECH_MS is not None:
                payload["request"]["force_to_speech_time"] = int(ENV_FORCE_TO_SPEECH_MS)
            if ENV_VAD_SEGMENT_MS is not None:
                payload["request"]["vad_segment_duration"] = int(ENV_VAD_SEGMENT_MS)
            if ENV_ENABLE_ACCELERATE is not None:
                payload["request"]["enable_accelerate_text"] = bool(ENV_ENABLE_ACCELERATE)
            if ENV_ACCELERATE_SCORE is not None:
                payload["request"]["accelerate_score"] = int(ENV_ACCELERATE_SCORE)
            # language is only supported on nostream endpoint; do not set here for bigmodel_async

            comp = self._gzip(json.dumps(payload).encode("utf-8"))
            return b"".join([hdr, struct.pack(">i", seq), struct.pack(">I", len(comp)), comp])

        def _full_client_request_noseq(self) -> bytes:
            MT_CLIENT_FULL = 0b0001
            MTS_NO_SEQ = 0b0000
            SER_JSON = 0b0001
            CMP_GZIP = 0b0001

            hdr = self._header_bytes(MT_CLIENT_FULL, MTS_NO_SEQ, SER_JSON, CMP_GZIP)
            payload = {
                "user": {"uid": "livekit_stt"},
                "audio": {
                    "format": "pcm",
                    "codec": "raw",
                    "rate": self._parent._opts.sample_rate,
                    "bits": 16,
                    "channel": 1,
                },
                "request": {
                    "model_name": "bigmodel",
                    "enable_itn": True,
                    "enable_punc": True,
                    "enable_ddc": True,
                    "show_utterances": True,
                    "result_type": ENV_RESULT_TYPE,
                },
            }
            if ENV_ENABLE_NONSTREAM is None:
                payload["request"]["enable_nonstream"] = True
            else:
                payload["request"]["enable_nonstream"] = bool(ENV_ENABLE_NONSTREAM)
            if ENV_END_WINDOW_MS is not None:
                payload["request"]["end_window_size"] = int(ENV_END_WINDOW_MS)
            if ENV_FORCE_TO_SPEECH_MS is not None:
                payload["request"]["force_to_speech_time"] = int(ENV_FORCE_TO_SPEECH_MS)
            if ENV_VAD_SEGMENT_MS is not None:
                payload["request"]["vad_segment_duration"] = int(ENV_VAD_SEGMENT_MS)
            if ENV_ENABLE_ACCELERATE is not None:
                payload["request"]["enable_accelerate_text"] = bool(ENV_ENABLE_ACCELERATE)
            if ENV_ACCELERATE_SCORE is not None:
                payload["request"]["accelerate_score"] = int(ENV_ACCELERATE_SCORE)
            # language only for nostream; not set here
            comp = self._gzip(json.dumps(payload).encode("utf-8"))
            return b"".join([hdr, struct.pack(">I", len(comp)), comp])

        def _audio_only_request(self, seq: int, chunk: bytes, is_last: bool) -> bytes:
            MT_CLIENT_AUDIO = 0b0010
            MTS_POS_SEQ = 0b0001
            MTS_LAST_NO_SEQ = 0b0010
            SER_NONE = 0b0000
            CMP_GZIP = 0b0001

            flags = MTS_LAST_NO_SEQ if is_last else MTS_POS_SEQ
            # Audio frames are raw PCM payload; mark serialization as NONE
            hdr = self._header_bytes(MT_CLIENT_AUDIO, flags, SER_NONE, CMP_GZIP)
            comp = self._gzip(chunk)
            if is_last:
                return b"".join([hdr, struct.pack(">I", len(comp)), comp])
            else:
                return b"".join([hdr, struct.pack(">i", seq), struct.pack(">I", len(comp)), comp])

        def _parse_server_response(self, data: bytes) -> tuple[bool, dict[str, Any]]:
            header_size_words = data[0] & 0x0F
            message_type = data[1] >> 4
            flags = data[1] & 0x0F
            serialization = data[2] >> 4
            compression = data[2] & 0x0F
            payload = data[header_size_words * 4 :]

            if flags & 0x01:
                payload = payload[4:]
            is_last = bool(flags & 0x02)
            if flags & 0x04:
                payload = payload[4:]

            err_code = None
            size = 0
            if message_type == 0xF and len(payload) >= 8:
                err_code = struct.unpack(">i", payload[:4])[0]
                size = struct.unpack(">I", payload[4:8])[0]
                payload = payload[8:]
            elif len(payload) >= 4:
                size = struct.unpack(">I", payload[:4])[0]
                payload = payload[4:]

            if compression == 0x1:
                try:
                    payload = __import__("gzip").decompress(payload)
                except Exception:
                    pass

            obj: dict[str, Any] = {}
            if serialization == 0x1 and size:
                try:
                    obj = json.loads(payload.decode("utf-8", errors="ignore"))
                except Exception:
                    obj = {}

            if err_code is not None and err_code != 0:
                # Check for recoverable errors that should trigger reconnection
                error_msg = obj.get('error', '') if isinstance(obj, dict) else ''

                # Retryable error codes:
                # 45000081: timeout waiting for next packet
                # 45000292: quota exceeded for audio_duration_lifetime
                is_retryable = True

                raise APIStatusError(
                    message=f"server error: code={err_code}, msg={error_msg}",
                    status_code=err_code,
                    request_id=self._request_id or None,
                    body=obj,
                    retryable=is_retryable,
                )

            return is_last, obj

        def _extract_text_and_times(self, obj: dict[str, Any]) -> tuple[str | None, float, float]:
            if not isinstance(obj, dict):
                return None, 0.0, 0.0
            res = obj.get("result")
            if isinstance(res, dict):
                txt = res.get("text")
                if isinstance(txt, str) and txt.strip():
                    # extract times from utterances if present
                    uts = res.get("utterances")
                    if isinstance(uts, list) and uts:
                        try:
                            st = min(float(u.get("start_time", 0)) for u in uts if isinstance(u, dict)) / 1000.0
                            et = max(float(u.get("end_time", 0)) for u in uts if isinstance(u, dict)) / 1000.0
                        except Exception:
                            st = et = 0.0
                    else:
                        st = et = 0.0
                    return txt, st, et
                # try building from utterances text or additions.fixed_prefix_result
                uts = res.get("utterances")
                if isinstance(uts, list) and uts:
                    try:
                        parts = [u.get("text", "") for u in uts if isinstance(u, dict)]
                        if not any(parts):
                            # fallback to additions.fixed_prefix_result
                            alt_parts = []
                            for u in uts:
                                if isinstance(u, dict):
                                    addi = u.get("additions")
                                    if isinstance(addi, dict):
                                        fx = addi.get("fixed_prefix_result")
                                        if isinstance(fx, str) and fx:
                                            alt_parts.append(fx)
                            if alt_parts:
                                parts = alt_parts
                        txt2 = "".join(t for t in parts if t)
                        st = min(float(u.get("start_time", 0)) for u in uts if isinstance(u, dict)) / 1000.0
                        et = max(float(u.get("end_time", 0)) for u in uts if isinstance(u, dict)) / 1000.0
                        if txt2.strip():
                            return txt2, st, et
                    except Exception:
                        pass
            # fallbacks
            for k in ("text", "transcript", "caption"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v, 0.0, 0.0
            return None, 0.0, 0.0

        async def _run(self) -> None:
            # Connect WS
            opts = self._parent._opts
            session = self._parent._ensure_session()
            headers = {
                "X-Api-Resource-Id": opts.resource_id or os.environ.get("DOUBAO_STT_RESOURCE_ID", "volc.bigasr.sauc.duration"),
                "X-Api-Request-Id": str(uuid.uuid4()),
                "X-Api-Access-Key": opts.access_token,
                "X-Api-App-Key": opts.app_id,
                "X-Api-Connect-Id": str(uuid.uuid4()),
            }
            try:
                self._ws = await session.ws_connect(
                    opts.endpoint,
                    headers=headers,
                    timeout=self._conn_options.timeout,
                    max_msg_size=10 * 1024 * 1024,
                )
            except asyncio.TimeoutError as e:
                raise APITimeoutError() from e
            except Exception as e:
                raise APIConnectionError("could not connect to Doubao STT (stream)") from e

            # capture X-Tt-Logid if present
            try:
                self._request_id = getattr(self._ws, "_response").headers.get("X-Tt-Logid", "")
            except Exception:
                self._request_id = ""

            # Prefer FULL handshake with sequence for reliable end-of-stream semantics
            await self._ws.send_bytes(self._full_client_request(1))
            # audio sequence starts at 2
            self._seq = 2

            # Receiver task: parse server responses and emit events
            async def _receiver() -> None:
                in_speech = False
                while self._running and self._ws and not self._ws.closed:
                    try:
                        msg = await self._ws.receive()
                        self._last_activity_time = time.monotonic()  # Update activity timestamp
                    except asyncio.CancelledError:
                        # Normal cancellation during cleanup
                        break
                    except Exception as e:
                        logger.warning(f"websocket receive error: {e}")
                        break

                    if msg.type == aiohttp.WSMsgType.BINARY:
                        try:
                            is_last, obj = self._parse_server_response(msg.data)
                        except APIStatusError as e:
                            # APIStatusError will be caught by _main_task and retried if retryable=True
                            # Just re-raise it to let the retry mechanism handle it
                            logger.warning(
                                f"doubao stt error: code={e.status_code}, retryable={e.retryable}",
                                exc_info=e
                            )
                            raise
                        if os.environ.get("DOUBAO_STT_DEBUG") == "1":
                            try:
                                logger.info(f"doubao stt recv: is_last={is_last}, obj={json.dumps(obj, ensure_ascii=False)[:400]}")
                            except Exception:
                                pass
                        # Prefer utterance-based incremental emission to avoid duplicate text
                        res = obj.get("result") if isinstance(obj, dict) else None
                        uts = res.get("utterances") if isinstance(res, dict) else None
                        # Emit start-of-speech when we see first text
                        any_text = False
                        try:
                            if isinstance(uts, list) and uts:
                                # any textual content in current frame
                                for u in uts:
                                    if isinstance(u, dict) and isinstance(u.get("text"), str) and u.get("text").strip():
                                        any_text = True
                                        break
                            else:
                                t0, _, _ = self._extract_text_and_times(obj)
                                any_text = bool(t0)
                        except Exception:
                            any_text = False

                        if any_text and not in_speech:
                            in_speech = True
                            self._speaking = True
                            ev = stt.SpeechEvent(type=SpeechEventType.START_OF_SPEECH, request_id=self._request_id)
                            try:
                                self._event_ch.send_nowait(ev)
                            except Exception:
                                await self._event_ch.send(ev)

                        if isinstance(uts, list) and uts:
                            # 1) Emit new FINALs only for newly-definite utterances
                            new_definite = 0
                            for i, u in enumerate(uts):
                                if not isinstance(u, dict):
                                    continue
                                if bool(u.get("definite")):
                                    idx = i + 1  # count of definite utterances up to i
                                    if idx > self._emitted_definite_count:
                                        txt = u.get("text") or ""
                                        if isinstance(txt, str) and txt.strip():
                                            ev = stt.SpeechEvent(
                                                type=SpeechEventType.FINAL_TRANSCRIPT,
                                                request_id=self._request_id,
                                                alternatives=[
                                                    stt.SpeechData(
                                                        text=txt,
                                                        language=self._parent._opts.language_code,
                                                        start_time=float(u.get("start_time", 0)) / 1000.0,
                                                        end_time=float(u.get("end_time", 0)) / 1000.0,
                                                    )
                                                ],
                                            )
                                            try:
                                                self._event_ch.send_nowait(ev)
                                            except Exception:
                                                await self._event_ch.send(ev)
                                            new_definite += 1
                            if new_definite:
                                self._emitted_definite_count += new_definite
                                # End of this speech segment
                                if in_speech:
                                    in_speech = False
                                    self._speaking = False
                                    # flush usage metrics for this segment
                                    if self._usage_accum > 0.0:
                                        usage_ev = stt.SpeechEvent(
                                            type=SpeechEventType.RECOGNITION_USAGE,
                                            recognition_usage=stt.RecognitionUsage(audio_duration=self._usage_accum),
                                            request_id=self._request_id,
                                        )
                                        try:
                                            self._event_ch.send_nowait(usage_ev)
                                        except Exception:
                                            await self._event_ch.send(usage_ev)
                                        self._usage_accum = 0.0
                                        self._last_usage_emit = time.monotonic()
                                    ev = stt.SpeechEvent(type=SpeechEventType.END_OF_SPEECH, request_id=self._request_id)
                                    try:
                                        self._event_ch.send_nowait(ev)
                                    except Exception:
                                        await self._event_ch.send(ev)

                            # 2) Emit INTERIM for the current unfinished utterance only if it changed
                            if len(uts) > self._emitted_definite_count:
                                cur = uts[-1]
                                if isinstance(cur, dict) and not bool(cur.get("definite")):
                                    itxt = cur.get("text") or ""
                                    if isinstance(itxt, str) and itxt != self._last_interim_text and itxt.strip():
                                        self._last_interim_text = itxt
                                        ev = stt.SpeechEvent(
                                            type=SpeechEventType.INTERIM_TRANSCRIPT,
                                            request_id=self._request_id,
                                            alternatives=[
                                                stt.SpeechData(
                                                    text=itxt,
                                                    language=self._parent._opts.language_code,
                                                    start_time=float(cur.get("start_time", 0)) / 1000.0,
                                                    end_time=float(cur.get("end_time", 0)) / 1000.0,
                                                )
                                            ],
                                        )
                                        try:
                                            self._event_ch.send_nowait(ev)
                                        except Exception:
                                            await self._event_ch.send(ev)
                        else:
                            # Fallback: no utterances — emit INTERIM deltas and FINAL on last
                            text, st, et = self._extract_text_and_times(obj)
                            if text:
                                if is_last:
                                    # compute delta over previously finalized text
                                    delta = text[len(self._final_accum_text) :] if text.startswith(self._final_accum_text) else text
                                    if delta.strip():
                                        self._final_accum_text = text
                                        ev = stt.SpeechEvent(
                                            type=SpeechEventType.FINAL_TRANSCRIPT,
                                            request_id=self._request_id,
                                            alternatives=[
                                                stt.SpeechData(
                                                    text=delta,
                                                    language=self._parent._opts.language_code,
                                                    start_time=st,
                                                    end_time=et,
                                                )
                                            ],
                                        )
                                        try:
                                            self._event_ch.send_nowait(ev)
                                        except Exception:
                                            await self._event_ch.send(ev)
                                        # finalize this segment
                                        if in_speech:
                                            in_speech = False
                                            self._speaking = False
                                            if self._usage_accum > 0.0:
                                                usage_ev = stt.SpeechEvent(
                                                    type=SpeechEventType.RECOGNITION_USAGE,
                                                    recognition_usage=stt.RecognitionUsage(audio_duration=self._usage_accum),
                                                    request_id=self._request_id,
                                                )
                                                try:
                                                    self._event_ch.send_nowait(usage_ev)
                                                except Exception:
                                                    await self._event_ch.send(usage_ev)
                                                self._usage_accum = 0.0
                                                self._last_usage_emit = time.monotonic()
                                            ev2 = stt.SpeechEvent(type=SpeechEventType.END_OF_SPEECH, request_id=self._request_id)
                                            try:
                                                self._event_ch.send_nowait(ev2)
                                            except Exception:
                                                await self._event_ch.send(ev2)
                                else:
                                    if text != self._last_interim_text:
                                        self._last_interim_text = text
                                        ev = stt.SpeechEvent(
                                            type=SpeechEventType.INTERIM_TRANSCRIPT,
                                            request_id=self._request_id,
                                            alternatives=[
                                                stt.SpeechData(
                                                    text=text,
                                                    language=self._parent._opts.language_code,
                                                    start_time=st,
                                                    end_time=et,
                                                )
                                            ],
                                        )
                                        try:
                                            self._event_ch.send_nowait(ev)
                                        except Exception:
                                            await self._event_ch.send(ev)

                        # Emit end-of-speech when server marks last while we are closing
                        if (is_last and self._closing) and in_speech:
                            in_speech = False
                            self._speaking = False
                            ev = stt.SpeechEvent(type=SpeechEventType.END_OF_SPEECH, request_id=self._request_id)
                            try:
                                self._event_ch.send_nowait(ev)
                            except Exception:
                                await self._event_ch.send(ev)
                            break
                    elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                        code = self._ws.close_code
                        reason = self._ws.close_message
                        logger.warning(f"doubao stt connection closed by server (code={code}, reason={reason})")
                        break

            recv_task = asyncio.create_task(_receiver())

            # Sender loop: buffer frames to ~200ms chunks; no keepalive silence
            send_error = None  # Track sender errors to propagate after cleanup
            bytes_per_sec = self._parent._opts.sample_rate * 2  # mono int16
            seg_ms = int(ENV_SEG_MS)
            target_chunk = max(1, (bytes_per_sec * seg_ms) // 1000)
            buf = bytearray()
            sent_seconds = 0.0

            async def send_chunk(chunk: bytes, *, is_last: bool = False, is_silence: bool = False) -> None:
                nonlocal sent_seconds
                if self._ws and not self._ws.closed:
                    pkt = self._audio_only_request(self._seq, chunk, is_last=is_last)
                    await self._ws.send_bytes(pkt)
                    self._seq += 1
                    # accumulate usage metrics only while speaking, and throttle
                    try:
                        if not is_silence and self._speaking:
                            dur = len(chunk) / float(self._parent._opts.sample_rate * 2)
                            self._usage_accum += dur
                            now = time.monotonic()
                            if self._usage_accum > 0.0 and (now - self._last_usage_emit) >= self._usage_period:
                                usage_ev = stt.SpeechEvent(
                                    type=SpeechEventType.RECOGNITION_USAGE,
                                    recognition_usage=stt.RecognitionUsage(audio_duration=self._usage_accum),
                                    request_id=self._request_id,
                                )
                                # non-blocking usage update
                                try:
                                    self._event_ch.send_nowait(usage_ev)
                                except Exception:
                                    await self._event_ch.send(usage_ev)
                                self._usage_accum = 0.0
                                self._last_usage_emit = now
                    except Exception:
                        pass

            already_sent_last = False
            try:
                async for item in self._input_ch:
                    # Check if receiver task failed
                    if recv_task.done():
                        exc = recv_task.exception()
                        if exc:
                            logger.warning(f"receiver task failed, stopping sender: {exc}")
                            raise exc

                    if isinstance(item, stt.RecognizeStream._FlushSentinel):
                        # End of current segment: flush any buffered audio.
                        if buf:
                            chunk = bytes(buf)
                            buf.clear()
                            await send_chunk(chunk, is_last=False, is_silence=False)
                        # continue to next segment without closing the stream
                        continue
                    frame: rtc.AudioFrame = item  # type: ignore[assignment]
                    try:
                        # send raw PCM int16 bytes
                        b = frame.data.cast("B").tobytes()
                    except Exception:
                        # robust fallback: combine and extract bytes
                        try:
                            combined = rtc.combine_audio_frames([frame])
                            b = combined.data.cast("B").tobytes()
                        except Exception:
                            continue
                    buf.extend(b)
                    while len(buf) >= target_chunk:
                        chunk = bytes(buf[:target_chunk])
                        del buf[:target_chunk]
                        await send_chunk(chunk, is_last=False, is_silence=False)
            except asyncio.CancelledError:
                # Normal cancellation during cleanup
                logger.debug("sender loop cancelled")
                pass
            except Exception as e:
                logger.warning(f"doubao stt sender error: {e}")
                send_error = e  # Save error to re-raise after cleanup
            finally:
                # This finally block runs when input channel is closed (end_input/aclose)
                # Flush any remaining buffer, then send the terminal packet to finalize.
                if self._ws and not self._ws.closed:
                    if buf and not already_sent_last:
                        try:
                            await send_chunk(bytes(buf), is_last=False, is_silence=False)
                        except Exception:
                            pass
                    if not already_sent_last:
                        try:
                            self._closing = True
                            await send_chunk(b"", is_last=True, is_silence=True)
                            already_sent_last = True
                        except Exception:
                            pass
                    # flush any pending usage
                    try:
                        if self._usage_accum > 0.0:
                            usage_ev = stt.SpeechEvent(
                                type=SpeechEventType.RECOGNITION_USAGE,
                                recognition_usage=stt.RecognitionUsage(audio_duration=self._usage_accum),
                                request_id=self._request_id,
                            )
                            try:
                                self._event_ch.send_nowait(usage_ev)
                            except Exception:
                                await self._event_ch.send(usage_ev)
                            self._usage_accum = 0.0
                            self._last_usage_emit = time.monotonic()
                    except Exception:
                        pass
                self._running = False
                # Wait for receiver task to complete and get its result
                recv_error = None
                try:
                    await recv_task
                except Exception as e:
                    recv_error = e
                    logger.warning(f"receiver task raised exception: {e}")
                finally:
                    # Always close the WebSocket
                    if self._ws:
                        try:
                            await self._ws.close()
                        except Exception as e:
                            logger.debug(f"error closing websocket: {e}")

                # Re-raise receiver error (takes precedence over sender error)
                if recv_error:
                    raise recv_error
                # Re-raise sender error if no receiver error
                if send_error:
                    raise send_error

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        if is_given(language):
            self._opts.language_code = language
        # Force sample rate to expected by SAUC
        return STT._SAUCStream(stt=self, conn_options=conn_options, sample_rate=self._opts.sample_rate)

    def with_vad(self, vad: Any) -> stt.STT:
        """Wrap this STT with LiveKit's StreamAdapter using the provided VAD.

        This is useful when you want client-side VAD-driven segmentation even if
        the remote STT service doesn't emit reliable start/end-of-speech events.
        """
        return stt.StreamAdapter(stt=self, vad=vad)

    @classmethod
    def with_vad_factory(cls, vad: Any, **kwargs: Any) -> stt.STT:
        """Convenience factory to build a Doubao STT wrapped with a VAD adapter.

        Example:
          vad = silero.VAD.load(min_silence_duration=0.75)
          stt = doubao.STT.with_vad_factory(vad, language_code="zh")
        """
        return stt.StreamAdapter(stt=cls(**kwargs), vad=vad)
