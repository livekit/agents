# livekit-plugins-gnani

[![PyPI](https://img.shields.io/pypi/v/livekit-plugins-gnani)](https://pypi.org/project/livekit-plugins-gnani/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[LiveKit Agents](https://github.com/livekit/agents) plugin for **[Gnani Vachana](https://gnani.ai/)** — high-accuracy Speech-to-Text and low-latency Text-to-Speech for Indian languages.

> **Vachana** is a production-ready speech AI platform by [Gnani.ai](https://gnani.ai) supporting 10+ Indian languages with real-time streaming, multilingual transcription, and code-switching capabilities.

## Installation

```bash
pip install livekit-plugins-gnani
```

This will also install the [`gnani-vachana`](https://pypi.org/project/gnani-vachana/) core SDK as a dependency.

## Prerequisites

You need a Gnani API key. Email **[speechstack@gnani.ai](mailto:speechstack@gnani.ai)** to get started — all new accounts receive free credits, no credit card required.

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
```

### Text-to-Speech

```python
from livekit.plugins.gnani import TTS

tts = TTS(voice="sia")
```

## Available Voices

| Voice | ID |
| --- | --- |
| Sia | `sia` |
| Raju | `raju` |
| Kanika | `kanika` |
| Nikita | `nikita` |
| Ravan | `ravan` |
| Simran | `simran` |
| Karan | `karan` |
| Neha | `neha` |

## Supported Languages

| Language | Code |
| --- | --- |
| Bengali | `bn-IN` |
| English (India) | `en-IN` |
| Gujarati | `gu-IN` |
| Hindi | `hi-IN` |
| Kannada | `kn-IN` |
| Malayalam | `ml-IN` |
| Marathi | `mr-IN` |
| Punjabi | `pa-IN` |
| Tamil | `ta-IN` |
| Telugu | `te-IN` |

## Architecture

```
gnani-vachana           ← Core SDK (REST, WebSocket, SSE clients)
    ↑
livekit-plugins-gnani   ← This package (LiveKit Agents adapter)
```

## Documentation

- [Vachana API Docs](https://docs.inya.ai/vachana/introduction/introduction)
- [LiveKit Agents Docs](https://docs.livekit.io/agents/)
- [gnani-vachana SDK](https://pypi.org/project/gnani-vachana/)

## License

Apache 2.0 — see [LICENSE](LICENSE).
