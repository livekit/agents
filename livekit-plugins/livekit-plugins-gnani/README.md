# livekit-plugins-gnani

[![PyPI](https://img.shields.io/pypi/v/livekit-plugins-gnani)](https://pypi.org/project/livekit-plugins-gnani/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

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
        recognize_method="websocket",
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

Swap `recognize_method` or `synthesize_method` for REST variants — see below. WebSocket STT + TTS is the default for lowest latency.

## Service Construction

### Speech-to-Text (REST)

```python
from livekit.plugins.gnani import STT

stt = STT(
    language="hi-IN",
    recognize_method="rest",
)
```

REST mode requires a VAD in the pipeline. LiveKit wraps the STT with `stt.StreamAdapter` automatically when `recognize_method="rest"`.

### Speech-to-Text (Streaming WebSocket)

```python
from livekit.plugins.gnani import STT

stt = STT(
    language="hi-IN",
    recognize_method="websocket",
    sample_rate=16000,
)
```

### Text-to-Speech (REST)

```python
from livekit.plugins.gnani import TTS

tts = TTS(
    voice="Pranav",
    model="timbre-v2.0",
    synthesize_method="rest",
)
```

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
| REST | `recognize_method="rest"` | POST `/stt/v3` | File/buffer transcription. Requires VAD. |
| WebSocket | `recognize_method="websocket"` | `wss://api.vachana.ai/stt/v3/stream` | Real-time streaming with VAD. Default. |

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
    language="en-IN",              # Default: "en-IN"
    sample_rate=16000,             # Default: 16000 (also: 8000, 44100, 48000)
    format="verbatim",             # Default: "verbatim" (also: "transcribe" for ITN)
    itn_native_numerals=False,     # Default: False
    recognize_method="websocket",  # Default: "websocket" (also: "rest")
    api_key=None,                  # Default: None (reads GNANI_API_KEY env var)
    base_url="https://api.vachana.ai",
)
```

### TTS — All parameters

```python
from livekit.plugins.gnani import TTS

tts = TTS(
    voice="Pranav",                # Default: "Pranav" (timbre-v2.0: Kaveri, Shubhra, Deepak)
    model="timbre-v2.0",           # Default: "timbre-v2.0" (also: "timbre-v2.5" with 42 voices)
    language=None,                 # timbre-v2.5 only — e.g. "hi-IN", "en-IN"
    sample_rate=16000,             # Default: 16000 (also: 8000, 22050, 44100)
    encoding="linear_pcm",         # Default: "linear_pcm" (also: "oggopus")
    container="wav",               # Default: "wav" (also: "raw", "mp3", "mulaw", "ogg")
    num_channels=1,                # Default: 1
    bitrate=None,                  # Default: None (also: "96k", "128k", "192k")
    synthesize_method="rest",      # Default: "rest" (also: "sse", "websocket")
    api_key=None,                  # Default: None (reads GNANI_API_KEY env var)
    base_url="https://api.vachana.ai",
)
```

## Supported Languages

### STT Languages (Prisma)

STT uses BCP-47 locale codes (e.g. `hi-IN`, `bn-IN`).

For the full list of supported languages, see:

- **[STT REST — Supported Languages](https://docs.gnani.ai/api/STT/speech-to-text#supported-languages)**
- **[STT Realtime — Supported Languages](https://docs.gnani.ai/api/STT/stt-websocket#supported-languages)**

### TTS Languages (Timbre)

The optional `language` parameter is supported for **`timbre-v2.5` only**. For the full list, see **[TTS — Supported Languages](https://docs.gnani.ai/api/TTS/tts-inference#supported-languages)**.

> **Migration:** The former model name `vachana-voice-v3` has been renamed to **`timbre-v2.0`**. Update any `model="vachana-voice-v3"` calls to `model="timbre-v2.0"` (or omit `model` to use the default).

## Available Voices

See the [official voice list](https://docs.gnani.ai/api/TTS/tts-sse#available-voices) for the latest supported voices.

### timbre-v2.0 (4 voices)

| Voice   | ID        | Gender | Description              |
|---------|-----------|--------|--------------------------|
| Pranav  | `Pranav`  | Male   | Bold, Trustworthy        |
| Kaveri  | `Kaveri`  | Female | Confident, Bright        |
| Shubhra | `Shubhra` | Female | Gentle, Expressive       |
| Deepak  | `Deepak`  | Male   | Grounded, Conversational |

### timbre-v2.5 (42 voices)

The expanded catalog includes voices across Hindi, English, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi, and Hinglish. Use `model="timbre-v2.5"` with an optional `language` parameter (e.g. `language="hi-IN"`).

```python
from livekit.plugins.gnani import TTS

tts = TTS(model="timbre-v2.5", voice="Nalini", language="hi-IN")
```

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

Apache 2.0 — see [LICENSE](LICENSE).
