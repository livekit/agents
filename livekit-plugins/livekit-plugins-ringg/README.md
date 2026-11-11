# Ringg plugin for LiveKit Agents

Support for [Ringg](https://ringg.ai/) streaming Speech-to-Text in LiveKit Agents.
Ringg STT is built for Indian-language voice agents: Hindi, Bengali, Gujarati,
Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu, plus English and Arabic.

## Installation

```bash
pip install livekit-plugins-ringg
```

Or with the `livekit-agents` extra:

```bash
pip install "livekit-agents[ringg]"
```

## Pre-requisites

You'll need a Ringg STT API key. Contact [Ringg](https://ringg.ai/) for API access.
Set it as an environment variable: `RINGG_API_KEY`.

## Usage

Use Ringg STT in an `AgentSession`:

```python
from livekit.agents import AgentSession
from livekit.plugins import ringg

session = AgentSession(
    stt=ringg.STT(language="hi"),
    # llm=..., tts=...
)
```

### Endpointing

By default the Ringg server detects the end of each user turn (`mode="stream"`):
interim transcripts stream in while the user speaks and a final transcript is
emitted when the server's VAD closes the turn.

For client-side endpointing, pass the same VAD you give the session:

```python
from livekit.plugins import ringg, silero

vad = silero.VAD.load()
session = AgentSession(
    vad=vad,
    stt=ringg.STT(language="hi", vad=vad),
)
```

The plugin then runs the VAD on the audio it receives, tells the server when the
user starts and stops speaking, and the server returns one final transcript per
turn (`mode="on_final"`). Set `mode=` explicitly to override either default.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `api_key` | `RINGG_API_KEY` | API key, if not set in the environment. |
| `language` | `"hi"` | Language code: `hi`, `bn`, `gu`, `kn`, `ml`, `mr`, `pa`, `ta`, `te`, `en`, `ar`. |
| `vad` | `None` | A LiveKit VAD. When given, switches to client-side endpointing. |
| `mode` | `None` | `"stream"` or `"on_final"`. Resolved from `vad` when not set. |
| `vad_tail_sil_ms` | `200` | Server VAD trailing silence in milliseconds. |
| `vad_confidence` | `0.55` | Server VAD confidence threshold. |
| `enable_cap_punc` | `True` | Server-side capitalisation and punctuation. |
| `base_url` | `None` | Ringg STT host override (`host[:port]`), for self-hosted deployments. |

The plugin is a thin layer over the [`ringglabs`](https://pypi.org/project/ringglabs/)
SDK, which owns the Ringg protocol and is documented there.
