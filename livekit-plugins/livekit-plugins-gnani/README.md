# livekit-plugins-gnani

[![PyPI](https://img.shields.io/pypi/v/livekit-plugins-gnani)](https://pypi.org/project/livekit-plugins-gnani/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[LiveKit Agents](https://github.com/livekit/agents) plugin for **[Gnani Vachana](https://gnani.ai/)** — high-accuracy Speech-to-Text and low-latency Text-to-Speech for Indian languages.

> **Vachana** is a production-ready speech AI platform by [Gnani.ai](https://gnani.ai) supporting 10+ Indian languages with real-time streaming, multilingual transcription, and code-switching capabilities.

## Installation

```bash
pip install livekit-plugins-gnani
```

This will also install the [`websockets`](https://pypi.org/project/websockets/) and [`livekit-agents`](https://pypi.org/project/livekit-agents/) packages as dependencies.

## Prerequisites

You need a Gnani API key. Email **[speechstack@gnani.ai](mailto:speechstack@gnani.ai)** to get started — all new accounts receive free credits, no credit card required.

Set your credentials as environment variables:

```bash
export GNANI_API_KEY="your-api-key"

# For REST STT only (optional):
export GNANI_ORGANIZATION_ID="your-org-id"
export GNANI_USER_ID="your-user-id"
```

## Quick Start

### Speech-to-Text

```python
from livekit.plugins.gnani import STT

stt = STT(language="hi-IN")

# Use with a LiveKit voice agent pipeline
```

### Text-to-Speech

```python
from livekit.plugins.gnani import TTS

tts = TTS(voice="Karan")

# Use with a LiveKit voice agent pipeline
```

## Features

### STT

- **Batch recognition** — REST API (`POST /stt/v3`) for file-based transcription
- **Real-time streaming** — WebSocket API for live audio transcription with VAD
- **10 Indian languages** — bn-IN, en-IN, gu-IN, hi-IN, kn-IN, ml-IN, mr-IN, pa-IN, ta-IN, te-IN
- **Code-switching** — Hinglish (en-hi-IN-latn) and Hindi-English mixed (en-hi-in-cm) for streaming
- **Sample rates** — 8 kHz and 16 kHz

### TTS

- **Chunked synthesis** — REST API for single-request audio generation
- **Real-time streaming** — WebSocket API for low-latency streaming synthesis
- **250+ voices** — v3 capitalized voices (Karan, Simran, Riya, etc.) + legacy v2 lowercase voices
- **Configurable output** — sample rate (8000–44100), encoding (linear_pcm, oggopus), container (raw, mp3, wav, mulaw, ogg)

## Supported Languages

| Language        | Code    |
|-----------------|---------|
| Bengali         | `bn-IN` |
| English (India) | `en-IN` |
| Gujarati        | `gu-IN` |
| Hindi           | `hi-IN` |
| Kannada         | `kn-IN` |
| Malayalam       | `ml-IN` |
| Marathi         | `mr-IN` |
| Punjabi         | `pa-IN` |
| Tamil           | `ta-IN` |
| Telugu          | `te-IN` |

## Available Voices

**v3 voices** (default model `vachana-voice-v3`, capitalized):

| Voice   | ID        | Language  |
|---------|-----------|-----------|
| Karan   | `Karan`   | Primary   |
| Simran  | `Simran`  | Primary   |
| Nara    | `Nara`    | Primary   |
| Riya    | `Riya`    | Primary   |
| Viraj   | `Viraj`   | Primary   |
| Raju    | `Raju`    | Primary   |

Plus 250+ language-specific voices for Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santhali, Sindhi, Tamil, Telugu, and Urdu.

**Legacy v2 voices** (model `vachana-voice-v2`, lowercase):

| Voice   | ID        |
|---------|-----------|
| sia     | `sia`     |
| raju    | `raju`    |
| kanika  | `kanika`  |
| nikita  | `nikita`  |
| ravan   | `ravan`   |
| simran  | `simran`  |
| karan   | `karan`   |
| neha    | `neha`    |

> **Note:** Casing matters — `"Karan"` uses v3 model, `"karan"` uses v2 model.

## Architecture

This plugin directly implements the Gnani Vachana REST and WebSocket APIs using `aiohttp` (for batch STT/TTS) and `websockets` (for streaming STT/TTS), adapting them into LiveKit's `stt.STT` and `tts.TTS` base classes. No external SDK is required — all connection logic, authentication, and audio format handling is self-contained.

## Documentation

- [Vachana API Docs](https://docs.inya.ai/vachana/introduction/introduction)
- [LiveKit Agents Docs](https://docs.livekit.io/agents/)

## License

Apache 2.0 — see [LICENSE](LICENSE).
