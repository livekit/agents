# livekit-plugins-gnani

[![PyPI](https://img.shields.io/pypi/v/livekit-plugins-gnani)](https://pypi.org/project/livekit-plugins-gnani/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[LiveKit Agents](https://github.com/livekit/agents) plugin for **[Gnani](https://gnani.ai/)** — high-accuracy Speech-to-Text (Prisma) and low-latency Text-to-Speech (Timbre) for Indian languages.

>[Gnani.ai](https://gnani.ai) featuring **Prisma** (STT) and **Timbre** (TTS) models, supporting 10+ Indian languages with real-time streaming, multilingual transcription, and code-switching capabilities.

## Installation

```bash
pip install livekit-plugins-gnani
```

This will also install the [`websockets`](https://pypi.org/project/websockets/) and [`livekit-agents`](https://pypi.org/project/livekit-agents/) packages as dependencies.

## Prerequisites

You need a Gnani API key from [app.gnani.ai/voice](https://app.gnani.ai/voice) (Gnani APIs).

### Authentication

All APIs require a single API key — no `organization_id` or `user_id` needed.

Set your credentials as environment variables:

```bash
export GNANI_API_KEY="your-api-key"
```

**Or pass the key in the constructor:**

```python
stt = STT(api_key="your-api-key", language="hi-IN")
tts = TTS(api_key="your-api-key")
```

> **Migration note:** If upgrading from an earlier version, remove any `organization_id` and `user_id` parameters — they are no longer accepted.

## Quick Start

### Speech-to-Text (REST + Streaming)

```python
from livekit.plugins.gnani import STT

stt = STT(language="hi-IN")

# REST STT (file-based transcription)
speech_event = await stt.recognize(audio_buffer)

# Streaming STT (real-time WebSocket)
speech_stream = stt.stream()
```

### Text-to-Speech

```python
from livekit.plugins.gnani import TTS

# REST (default) - single-request batch synthesis
tts_rest = TTS(voice="Pranav")

# SSE - chunked synthesis via Server-Sent Events (lower latency)
tts_sse = TTS(voice="Pranav", synthesize_method="sse")

# WebSocket - chunked synthesis over WS (lowest latency)
tts_ws = TTS(voice="Pranav", synthesize_method="websocket")
```

All three modes work with the standard LiveKit voice agent pipeline.
The `synthesize_method` controls which transport `synthesize()` uses
(REST, SSE, or WebSocket). The `stream()` method always uses WebSocket
regardless of this setting.

## Full Constructor Reference

### STT — All parameters

```python
from livekit.plugins.gnani import STT

stt = STT(
    language="en-IN",           # Default: "en-IN"
    sample_rate=16000,          # Default: 16000 (also: 8000)
    format="verbatim",          # Default: "verbatim" (also: "transcribe")
    preferred_language=None,    # Default: None
    itn_native_numerals=False,  # Default: False
    api_key=None,               # Default: None (reads GNANI_API_KEY env var)
    base_url="https://api.vachana.ai",  # Default
)
```

### TTS — All parameters

```python
from livekit.plugins.gnani import TTS

tts = TTS(
    voice="Pranav",                    # Default: "Pranav" (also: Kaveri, Shubhra, Deepak)
    model="vachana-voice-v3",         # Default: "vachana-voice-v3"
    sample_rate=16000,                # Default: 16000 (also: 8000, 22050, 44100)
    encoding="linear_pcm",           # Default: "linear_pcm" (also: "oggopus")
    container="wav",                  # Default: "wav" (also: "raw", "mp3", "mulaw", "ogg")
    num_channels=1,                   # Default: 1
    bitrate=None,                     # Default: None (also: "96k", "128k", "192k")
    synthesize_method="rest",         # Default: "rest" (also: "sse", "websocket")
    api_key=None,                     # Default: None (reads GNANI_API_KEY env var)
    base_url="https://api.vachana.ai",  # Default
)
```

## Features

### STT (Prisma)

- **REST recognition** — REST API (`POST /stt/v3`) for file-based transcription
- **Real-time streaming** — WebSocket API (`wss://api.vachana.ai/stt/v3/stream`) for live audio transcription with VAD
- **10+ Indian languages** — see [supported language codes](https://docs.gnani.ai/api/STT/stt-websocket#supported-languages)
- **Code-switching** — supports multilingual and code-mixed audio
- **Sample rates** — 8 kHz and 16 kHz
- **ITN support** — Inverse Text Normalization via `format="transcribe"`

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

### TTS (Timbre)

- **REST synthesis** — single-request batch audio generation (`synthesize_method="rest"`)
- **SSE streaming** — lower-latency chunked synthesis via Server-Sent Events (`synthesize_method="sse"`)
- **WebSocket synthesis** — lowest-latency synthesis via `synthesize_method="websocket"` or the `stream()` method
- **4 voices** — Pranav, Kaveri, Shubhra, Deepak (see [Available Voices](https://docs.gnani.ai/api/TTS/tts-sse#available-voices))
- **Configurable output** — sample rate (8000–44100), encoding (linear_pcm, oggopus), container (raw, mp3, wav, mulaw, ogg)
- **Runtime updates** — change voice or model via `update_options()`

## Supported Languages

### STT Languages (Prisma)

Prisma uses BCP-47 locale codes (e.g. `hi-IN`). Supported:

- **[STT REST — Supported Languages](https://docs.gnani.ai/api/STT/speech-to-text#supported-languages)**
- **[STT Realtime — Supported Languages](https://docs.gnani.ai/api/STT/stt-websocket#supported-languages)**

---

### TTS Languages (Timbre)

For the full list of supported languages, see **[TTS — Supported Languages](https://docs.gnani.ai/api/TTS/tts-inference#supported-languages)**.

## Available Voices

| Voice   | ID        | Gender | Description              |
|---------|-----------|--------|--------------------------|
| Pranav  | `Pranav`  | Male   | Bold, Trustworthy        |
| Kaveri  | `Kaveri`  | Female | Confident, Bright        |
| Shubhra | `Shubhra` | Female | Gentle, Expressive       |
| Deepak  | `Deepak`  | Male   | Grounded, Conversational |

## Architecture

This plugin directly implements the Gnani REST and WebSocket APIs using `aiohttp` (for REST STT/TTS) and `websockets` (for streaming STT/TTS), adapting them into LiveKit's `stt.STT` and `tts.TTS` base classes. It uses the **Prisma** model for speech-to-text and the **Timbre** model for text-to-speech. No external SDK is required — all connection logic, authentication, and audio format handling is self-contained. Authentication uses a single `api_key` passed via the `X-API-Key-ID` header.

## Documentation

- [Gnani API Docs](https://docs.gnani.ai/)
- [LiveKit Agents Docs](https://docs.livekit.io/agents/)
- [Gnani STT Plugin Guide](https://docs.livekit.io/agents/integrations/stt/gnani/)
- [Gnani TTS Plugin Guide](https://docs.livekit.io/agents/integrations/tts/gnani/)

## License

Apache 2.0 — see [LICENSE](LICENSE).
