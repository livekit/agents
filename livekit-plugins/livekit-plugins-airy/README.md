# Airy plugin for LiveKit Agents

Use the [Airy Cloud API](https://airy.so/cloud-api/docs/home) as a text-to-speech provider in LiveKit Agents. The plugin supports Korean and English synthesis with Airy's raw PCM streaming response.

## Architecture

Airy accepts one complete utterance per HTTP request. LiveKit's `StreamAdapter` converts streamed LLM text into sentences, while the plugin forwards Airy's response audio as soon as PCM bytes arrive.

```mermaid
flowchart LR
    LLM[LLM text stream] --> Adapter[LiveKit StreamAdapter]
    Adapter -->|complete sentence| AiryTTS[airy.TTS.synthesize]
    AiryTTS -->|POST JSON| API[Airy speech API]
    API -->|24 kHz s16le mono PCM| Emitter[LiveKit AudioEmitter]
    Emitter --> Frames[Audio frames]
```

| Capability | Behavior |
| --- | --- |
| Input streaming | Sentence-adapted by LiveKit (`streaming=False`) |
| HTTP output | Forwarded progressively without buffering the full response |
| Audio | Raw PCM, 24 kHz, signed 16-bit little-endian, mono |
| Languages | `ko`, `en` |
| Styles | `normal`, `bright`, `calm`, `whisper` |
| Word alignment | Not available |

## Installation

After this package is released:

```bash
pip install livekit-plugins-airy
```

For development from this repository:

```bash
uv sync --all-extras --dev
```

## Configuration

Set the Airy API key without printing it or committing it to source control:

```bash
export AIRY_API_KEY="your-api-key"
```

`language` is required because it controls synthesis and Airy billing. Other options use Airy's documented defaults.

| Option | Default | Notes |
| --- | --- | --- |
| `language` | Required | `ko` or `en` |
| `model` | `airy-tts-v1` | Airy model ID |
| `voice` | `a597bb7a98fc9ec1` | Airy voice ID |
| `style` | `normal` | `normal`, `bright`, `calm`, or `whisper` |
| `api_key` | `AIRY_API_KEY` | Explicit value takes precedence |
| `base_url` | `https://api.airy.so` | API root; do not include `/v1` |
| `http_session` | LiveKit shared session | An injected session remains caller-owned |

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import airy

session = AgentSession(
    # ... stt and llm configuration ...
    tts=airy.TTS(
        language="ko",
        model="airy-tts-v1",
        voice="a597bb7a98fc9ec1",
        style="normal",
    ),
)
```

An executable agent is available at [`examples/agent.py`](./examples/agent.py). It reads Airy and LiveKit credentials from environment variables:

```bash
uv run python livekit-plugins/livekit-plugins-airy/examples/agent.py dev
```

## Limits and retry behavior

- Each synthesized sentence must contain 1–1,280 Unicode characters. The plugin rejects empty, whitespace-only, and oversized input without truncating or splitting it.
- `streaming=False` describes text input. The HTTP audio response is still delivered progressively.
- HTTP 400, 401, 402, 403, and 404 responses are not retried. Rate limits, server errors, timeouts, and connection failures may use LiveKit's retry policy before audio arrives.
- After any PCM bytes arrive, a timeout or connection failure is not retried. This avoids joining partial audio to a repeated synthesis and reduces duplicate billing risk.
- Airy's `Retry-After` header is retained as error metadata, but LiveKit's current retry scheduler does not dynamically honor it.
- Cancellation closes the client response, but it does not guarantee cancellation of server-side synthesis or billing.
