# SILMA AI plugin for LiveKit Agents

Support for Arabic and English speech synthesis with [SILMA AI](https://silma.ai/)
TTS v2 — Modern Standard Arabic, the Saudi (Najdi) dialect, and English.

More information is available in the docs for the
[TTS integration](https://docs.livekit.io/agents/models/tts/plugins/silma/).

## Installation

```bash
pip install livekit-plugins-silma
```

Or with the LiveKit Agents extra:

```bash
uv add "livekit-agents[silma]"
```

## Pre-requisites

You'll need an API key from [SILMA](https://app.silma.ai/api-keys). Set it as an
environment variable:

```bash
export SILMA_API_KEY="..."
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import silma

session = AgentSession(
    tts=silma.TTS(
        model="silma-tts-v2-msa",
        voice="maryam",
    ),
    # ... stt, llm, vad
)
```

### Models and voices

| Model | Language | Voices |
| --- | --- | --- |
| `silma-tts-v2-english` | English | `james`, `emma`, `isabella`, `oliver` |
| `silma-tts-v2-msa` | Modern Standard Arabic | `abdulaziz`, `fahd`, `layla`, `maryam`, `nouf`, `noura`, `omar`, `reema`, `yusuf`, `khalid`, `sarah`, `salma`, `salwa`, `saja`, `sultan`, `salman`, `sulaiman`, `salim` |
| `silma-tts-v2-ksa` | Arabic, Saudi (Najdi) dialect | same as MSA |

### Cloned voices

Upload a voice under **Custom Voices** at https://app.silma.ai/voices and pass
its id along with your user id:

```python
silma.TTS(
    model="silma-tts-v2-ksa",
    voice="maryam",
    user_id="...",
    custom_audio_id="voice_1769817467123",
)
```

### Pronunciation hints

SILMA reads phone numbers, emails and links correctly when they are tagged in
the text:

```python
await session.say(
    "You can reach us on <STAG_PN>92005455</STAG_PN> or at <STAG_EMAIL>hi@silma.ai</STAG_EMAIL>."
)
```

The plugin keeps these tags intact when it splits text, so a tag is never cut in
half across two requests.

Account-level pronunciation overrides configured at https://app.silma.ai/control
are applied when you pass `user_id` and
`enable_server_pronunciation_overrides=True`.

## How it works

`stream()` uses the realtime WebSocket API (`wss://api.silma.ai/tts/v2/ws/stream`)
and sentence-tokenizes incoming LLM text so each request is a complete
utterance. `synthesize()` uses the binary streaming endpoint
(`POST /stream`).

SILMA caps `text` at 250 characters per request, so longer input is split on
word boundaries and sent as sequential requests concatenated into one audio
segment.

SILMA returns a 24 kHz mono float32 waveform; the plugin converts it to 16-bit
PCM for the agent pipeline.

