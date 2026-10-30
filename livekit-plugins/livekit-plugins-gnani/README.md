# livekit-plugins-gnani

[![PyPI](https://img.shields.io/pypi/v/livekit-plugins-gnani)](https://pypi.org/project/livekit-plugins-gnani/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](../../LICENSE)

[LiveKit Agents](https://github.com/livekit/agents) plugin for **[Gnani](https://gnani.ai/)** — high-accuracy Speech-to-Text (Prisma) and low-latency Text-to-Speech (Timbre) for Indian languages.

> **Gnani** is a production-ready speech AI platform supporting 10+ Indian languages, real-time streaming, and multilingual transcription.

> This integration is maintained by [Gnani.ai](https://gnani.ai/).

## Installation

```bash
pip install livekit-plugins-gnani
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add livekit-plugins-gnani
```

This will also install the [`websockets`](https://pypi.org/project/websockets/) and [`livekit-agents`](https://pypi.org/project/livekit-agents/) packages as dependencies.

Install with the LiveKit Agents Gnani extra:

```bash
uv add "livekit-agents[gnani]"
```

## Prerequisites

You need a Gnani API key. [Gnani APIs](https://app.gnani.ai/voice) have this.

Set your credentials as environment variables:

```bash
export GNANI_API_KEY="your-api-key"
```

**Or pass the key in the constructor:**

```python
stt = STT(api_key="your-api-key", language="hi-IN")
tts = TTS(api_key="your-api-key")
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `GNANI_API_KEY` | API key for Gnani Vachana STT and TTS |

## Quick Start — AgentSession snippet

The snippet below shows core `AgentSession` wiring with Gnani WebSocket STT/TTS.

```python
from livekit.agents import AgentSession, room_io
from livekit.plugins import gnani, groq, silero

session = AgentSession(
    stt=gnani.STT(
        language="en-IN",
        use_streaming=True,
    ),
    llm=groq.LLM(model="llama-3.1-8b-instant"),
    tts=gnani.TTS(
        voice="Nalini",
        model="timbre-v2.5",
        language="en-IN",
        synthesize_method="websocket",
    ),
    vad=silero.VAD.load(),
)

await session.start(
    agent=MyAgent(),
    room=ctx.room,
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(sample_rate=16000),
        audio_output=room_io.AudioOutputOptions(sample_rate=16000),
    ),
)
```

Set `use_streaming=False` (STT) or swap `synthesize_method` (TTS) for REST variants — see below. WebSocket STT + TTS is the default for lowest latency.

## Service Construction

### Speech-to-Text (REST)

```python
from livekit.plugins.gnani import STT

stt = STT(
    language="hi-IN",
    use_streaming=False,
)
```

REST mode requires a VAD in the pipeline. LiveKit wraps the STT with `stt.StreamAdapter` automatically when `use_streaming=False`.

### Speech-to-Text (Streaming WebSocket)

```python
from livekit.plugins.gnani import STT

stt = STT(
    language="hi-IN",
    use_streaming=True,
    sample_rate=16000,
)
```

### Speech-to-Text (Prisma v3.0)

Prisma v3.0 is opt-in: pass `model="prisma-v3.0"`. Without `model`, the API default
(`prisma-v2.5`) is used and requests are unchanged. (The `3` in the `/stt/v3` URL is the API
version, not the model version.)

```python
from livekit.plugins.gnani import STT

stt = STT(
    model="prisma-v3.0",
    language="hi-IN",
    format="codemix",  # "verbatim" | "transcribe" | "codemix" | "transliterate"
    keyterms=["Vachana", "Prisma"],  # up to 200 terms to bias recognition towards
    topic="banking support call",  # optional context (<= 1600 characters)
    previous_text="Namaste, how can I help you?",  # optional; the last 600 characters are used
    speaker_names=["Customer", "Agent"],  # optional, 1 to 4 names
)
```

What Prisma v3.0 adds, and what the plugin does with it:

- **Several languages at once.** `language` can be a list, `["en-IN", "hi-IN"]` (up to 4), or
  `"indic-auto"` to detect among all Indic languages. Final transcripts carry the language the
  API detected (`detected_language_code`) instead of the configured one. A list needs
  `model="prisma-v3.0"`; on `prisma-v2.5` the plugin raises a `ValueError` at construction.
- **Formats.** `codemix` (Hindi/English mixed script) and `transliterate` are v3.0 only.
  `transliterate` responses carry no detected language, so the configured one is reported.
- **Keyterms and context.** `keyterms` also receive the terms LiveKit itself passes to the
  session (for example from `AgentSession(keyterms=...)`), merged and capped at 200.
  `topic`, `previous_text` and `speaker_names` are context hints for the model.
- **Tuning.** `idle_timeout_sec`, `vad_threshold`, `min_silence_ms`, `min_speech_ms`,
  `min_segment_ms` and `max_speech_s` set the server-side VAD/idle behaviour for the
  streaming session. Out-of-range values are rejected up front rather than silently clamped
  by the server.

#### Streaming behaviour

- The plugin advertises `manual_flush`: when LiveKit flushes the stream (for example at the end
  of a turn) the plugin sends a `commit` message so the server finalises the utterance
  immediately instead of waiting for its own silence detection. `commit` is also sent when the
  audio input ends, and the stream then keeps receiving until the last utterance's transcript
  (or error) arrives, however long the server takes, up to the connection timeout
  (`APIConnectOptions.timeout`). If that last chunk held no speech the server never answers, so
  the stream finishes once the timeout elapses.
- Each stream emits `START_OF_SPEECH`, then `FINAL_TRANSCRIPT`, then `END_OF_SPEECH`. The API
  sends one final transcript per utterance and no interim results. `END_OF_SPEECH` is therefore
  emitted right after the final transcript, which matters only with `turn_detection="stt"`.
- The server ends a session after a period without speech (`idle_timeout_sec`). The plugin
  reconnects transparently while the audio input is still open; this is not reported as an
  error. If the server keeps closing sessions immediately, the stream fails with a clear
  `APIConnectionError` instead of looping.
- A rejected handshake (bad key, unsupported option) is a non-retryable `APIStatusError` that
  includes the server's message. A per-utterance failure (for example unclear audio) is logged
  and the stream keeps running.
- `stt.update_options(language=..., format=..., keyterms=...)` applies to running streams
  through `update_settings`, without reconnecting. A multi-language list cannot be changed
  mid-session; it applies on the next reconnect.

> **Migration:** `language="en-IN,hi-IN"` (a comma-joined string) never worked against the API
> for streaming and failed on REST for `prisma-v2.5`. It is still accepted, and treated as a
> list, but now requires `model="prisma-v3.0"` (a `ValueError` otherwise). Streaming sends the
> API's `indic-auto` plus a candidate list, because the API rejects a comma list there.
> `preferred_language` is not part of Gnani's current documentation and is kept only for
> backwards compatibility.

### Text-to-Speech (REST)

```python
from livekit.plugins.gnani import TTS

tts = TTS(
    voice="Pranav",
    model="timbre-v2.5",
    synthesize_method="rest",
)
```

### Text-to-Speech (Timbre v3.0)

Timbre v3.0 has 13 voices that each speak all supported languages, and supports
[audio tags](https://docs.gnani.ai/api/TTS/tts-audio-tags) for expressive delivery.
Voices belong to a model, so pass a Timbre v3.0 voice (or omit `voice` to get `Avinash`).

```python
from livekit.plugins.gnani import TTS

tts = TTS(
    model="timbre-v3.0",
    voice="Shivani",
    language="hi-IN",  # optional; "auto" detects the language from the text
    speed=1.0,  # optional; 0.85 – 1.15
)
```

Audio tags are plain inline text, e.g. `"Haha <laugh> sach mein?"`. Tags are billed as
characters, and models other than Timbre v3.0 strip them before synthesis.

### Text-to-Speech (SSE Streaming)

```python
from livekit.plugins.gnani import TTS

tts = TTS(
    voice="Pranav",
    synthesize_method="sse",
)
```

### Text-to-Speech (WebSocket Streaming)

```python
from livekit.plugins.gnani import TTS

tts = TTS(
    voice="Nalini",
    model="timbre-v2.5",
    language="hi-IN",
    synthesize_method="websocket",
)
```

The `stream()` method always uses WebSocket regardless of `synthesize_method`.

## Services

### STT

| Mode | Parameter | Transport | Description |
|------|-----------|-----------|-------------|
| REST | `use_streaming=False` | POST `/stt/v3` | File/buffer transcription. Requires VAD. |
| WebSocket | `use_streaming=True` | `wss://api.vachana.ai/stt/v3/stream` | Real-time streaming with VAD. Default. |

#### Streaming PCM Specification

All streaming audio must be sent as **raw PCM binary frames** — no container format (WAV, MP3) mid-stream.

| Property          | 16 kHz                                    | 8 kHz                                     |
|-------------------|-------------------------------------------|-------------------------------------------|
| Encoding          | PCM signed 16-bit little-endian           | PCM signed 16-bit little-endian           |
| Sample Rate       | 16,000 Hz                                 | 8,000 Hz                                  |
| Channels          | 1 (mono)                                  | 1 (mono)                                  |
| Samples per chunk | 512                                       | 512                                       |
| **Bytes per frame** | **1,024 bytes** (512 samples × 2 bytes) | **1,024 bytes** (512 samples × 2 bytes)   |
| Frame duration    | 32 ms                                     | 64 ms                                     |

Frames must be sent at **real-time cadence**. See **[STT Realtime — PCM Specification](https://docs.gnani.ai/api/STT/stt-websocket#pcm-specification)** for full details.

### TTS

| Mode | Parameter | Transport | Description |
|------|-----------|-----------|-------------|
| REST | `synthesize_method="rest"` | POST `/api/v1/tts/inference` | Single-request batch synthesis. Default for `synthesize()`. |
| SSE | `synthesize_method="sse"` | POST `/api/v1/tts/sse` | Chunked synthesis via Server-Sent Events. |
| WebSocket | `synthesize_method="websocket"` or `stream()` | `wss://api.vachana.ai/api/v1/tts` | Lowest latency; `stream()` always uses WebSocket. |

## Full Constructor Reference

### STT — All parameters

```python
from livekit.plugins.gnani import STT

stt = STT(
    language="en-IN",  # str | list[str]; Default: "en-IN". Lists / "indic-auto" need prisma-v3.0
    model=None,  # Default: None (API default, prisma-v2.5); also "prisma-v3.0"
    sample_rate=16000,  # Default: 16000 (also: 8000, 44100, 48000)
    format="verbatim",  # "verbatim" | "transcribe" (ITN) | "codemix" | "transliterate" (v3.0 only)
    itn_native_numerals=False,  # Default: False (prisma-v2.5 only)
    use_streaming=True,  # Default: True (WebSocket); False = REST + VAD
    keyterms=None,  # up to 200 terms; prisma-v3.0 only
    topic=None,  # context, <= 1600 characters; prisma-v3.0 only
    previous_text=None,  # context, last 600 characters used; prisma-v3.0 only
    speaker_names=None,  # 1 to 4 names; prisma-v3.0 only
    idle_timeout_sec=None,  # streaming: server-side idle expiry (the plugin reconnects)
    vad_threshold=None,  # streaming: server-side VAD tuning (None = server default)
    min_silence_ms=None,
    min_speech_ms=None,
    min_segment_ms=None,
    max_speech_s=None,
    api_key=None,  # Default: None (reads GNANI_API_KEY env var)
    base_url="https://api.vachana.ai",
)
```

### TTS — All parameters

```python
from livekit.plugins.gnani import TTS

tts = TTS(
    model="timbre-v2.5",  # Default: "timbre-v2.5" (also: "timbre-v3.0")
    voice="Pranav",  # Default depends on model: "Pranav" (v2.5), "Avinash" (v3.0)
    language=None,  # e.g. "hi-IN", "en-IN", "hi-en", "auto"
    speed=None,  # 0.85 – 1.15; omitted from the request when unset
    sample_rate=16000,  # Default: 16000 (also: 8000, 22050, 24000, 44100, 48000)
    encoding="linear_pcm",  # Default: "linear_pcm" (also: "oggopus")
    container="wav",  # Default: "wav" (also: "raw", "mp3", "ogg")
    num_channels=1,  # Default: 1
    bitrate=None,  # Default: None (also: "96k", "128k", "192k")
    synthesize_method="rest",  # Default: "rest" (also: "sse", "websocket")
    api_key=None,  # Default: None (reads GNANI_API_KEY env var)
    base_url="https://api.vachana.ai",
)
```

## Supported Languages

### STT Languages (Prisma)

STT uses BCP-47 locale codes (e.g. `hi-IN`, `bn-IN`), plus `indic-auto` on `prisma-v3.0`.

For the full list of supported languages, see:

- **[STT REST — Supported Languages](https://docs.gnani.ai/api/STT/speech-to-text#supported-languages)**
- **[STT Realtime — Supported Languages](https://docs.gnani.ai/api/STT/stt-websocket#supported-languages)**

### TTS Languages (Timbre)

The optional `language` parameter is supported by **`timbre-v2.5`** and **`timbre-v3.0`**. For the full list, see **[TTS — Supported Languages](https://docs.gnani.ai/api/TTS/tts-inference#supported-languages)**.

> **Migration:** `timbre-v2.0` has been **retired by the Gnani API** (requests fail with `400 ... is no longer available`) and is rejected with a `ValueError` as soon as you construct `TTS(model="timbre-v2.0")`. The default model is now `timbre-v2.5`. Voices are model-specific, so if you used a `timbre-v2.0`-only voice such as `Shubhra`, choose one from the [voice list](https://docs.gnani.ai/api/TTS/available-voices).

## Available Voices

See the [official voice list](https://docs.gnani.ai/api/TTS/tts-sse#available-voices) for the latest supported voices.

### timbre-v2.5

The expanded catalog covers Hindi, English, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi, and Hinglish — for example `Pranav` (default), `Kaveri`, `Deepak`, `Nalini`, `Poorvi`. Use `model="timbre-v2.5"` with an optional `language` parameter (e.g. `language="hi-IN"`).

```python
from livekit.plugins.gnani import TTS

tts = TTS(model="timbre-v2.5", voice="Nalini", language="hi-IN")
```

### timbre-v3.0 (13 voices)

`Avinash` (default), `Chetana`, `Devansh`, `Harshad`, `Kalpana`, `Lakshmi`, `Prakash`, `Riyaa`, `Shivani`, `Shubham`, `Tarika`, `Vasudha`, `Yashraj`. Every voice speaks all supported languages and supports audio tags.

### Voice validation

Voices are bound to a model. The plugin raises `ValueError` at construction (or in `update_options`) when a voice is known to belong to a *different* model — for example `TTS(model="timbre-v3.0", voice="Pranav")` — instead of failing on the first synthesis with an API `400`. Voice names the plugin does not know are passed through unchanged, so newly added voices work without a plugin update; the API remains the source of truth. When only `model` changes in `update_options()` and you never chose a voice, the voice follows the new model's default.

## Architecture

```
livekit-plugins-gnani    ← This package (LiveKit Agents adapter)
  ├── STT: REST + WebSocket
  └── TTS: REST + SSE + WebSocket
```

This plugin directly implements the Gnani REST and WebSocket APIs using `aiohttp` (for REST STT/TTS) and `websockets` (for streaming STT/TTS), adapting them into LiveKit's `stt.STT` and `tts.TTS` base classes. It uses the **Prisma** model for speech-to-text and the **Timbre** model for text-to-speech. No external SDK is required — all connection logic, authentication, and audio format handling is self-contained. Authentication uses a single `api_key` passed via the `X-API-Key-ID` header.

## Documentation

- [Gnani API Docs](https://docs.gnani.ai/)
- [LiveKit Agents Docs](https://docs.livekit.io/agents/)
- [Gnani STT Plugin Guide](https://docs.livekit.io/agents/integrations/stt/gnani/)
- [Gnani TTS Plugin Guide](https://docs.livekit.io/agents/integrations/tts/gnani/)
- [STT REST API](https://docs.gnani.ai/api/STT/speech-to-text)
- [STT Realtime WebSocket](https://docs.gnani.ai/api/STT/stt-websocket)
- [TTS REST API](https://docs.gnani.ai/api/TTS/tts-inference)
- [TTS Streaming (SSE)](https://docs.gnani.ai/api/TTS/tts-sse)
- [TTS Realtime WebSocket](https://docs.gnani.ai/api/TTS/tts-websocket)

## LiveKit Compatibility

Tested with **LiveKit Agents v1.6.x**.

## License

Apache 2.0 — see [LICENSE](../../LICENSE).
