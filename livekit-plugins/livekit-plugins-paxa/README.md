# Paxa plugin for LiveKit Agents

Support for text-to-speech with [Paxa](https://paxalabs.com). Every voice on Paxa TTS
Flash v2 speaks Thai, English and Mandarin, including sentences that mix them.

See [https://paxalabs.com/docs/text-to-speech](https://paxalabs.com/docs/text-to-speech)
for provider docs.

## Installation

```bash
pip install livekit-plugins-paxa
```

Or with the LiveKit Agents extra:

```bash
uv add "livekit-agents[paxa]"
```

## Pre-requisites

You'll need an API key from [Paxa](https://paxalabs.com/app/keys). Set it as an
environment variable:

```bash
export PAXA_API_KEY="pxa_..."
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import paxa

session = AgentSession(
    tts=paxa.TTS(
        model="paxa-tts-flash-v2",
        voice="foithong",
        language="auto",
        speed=1.0,
        sample_rate=24000,
    ),
    # ... stt, llm, vad
)
```

`voice` takes any voice id from
[`GET /v1/voices`](https://paxalabs.com/docs/api/voices), such as `foithong` or
`massaman`, or a designed voice id (`voice_...`). `language` sets how digits, codes and
symbols are read (`auto`, `th`, `en` or `zh`); words keep the language of their script,
so mixed text needs no setting. `speed` runs from 0.5 to 1.5. `sample_rate` is the output
rate in Hz: Paxa TTS Flash v2 serves 8000, 16000, 22050, 24000, 32000, 44100 and 48000,
and Paxa TTS Flash 24000 only. Every rate costs the same.

`stream()` uses the [live WebSocket](https://paxalabs.com/docs/live-streaming)
(`/v1/tts/live`). Each stream opens one session when its first text arrives and sends
text one sentence at a time, split by `tokenizer` (LiveKit's sentence tokenizer by
default). Sentences that arrive while a sentence is still being synthesized are sent
together. On interruption, the plugin closes the session's socket, which stops synthesis
right away; the next turn opens a new session.

`synthesize()` streams a single `POST /v1/tts` request.

Both return 16-bit mono PCM at `sample_rate`, 24 kHz by default. An open live session
holds one of your plan's concurrent speech slots.
