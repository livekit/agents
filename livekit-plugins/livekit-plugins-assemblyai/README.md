# AssemblyAI plugin for LiveKit Agents

Support for Streaming Speech-to-Text and Streaming Text-to-Speech from AssemblyAI.

See [https://docs.livekit.io/agents/integrations/stt/assemblyai/](https://docs.livekit.io/agents/integrations/stt/assemblyai/) for more information.

## Installation

```bash
pip install livekit-plugins-assemblyai
```

## Pre-requisites

You'll need to specify an AssemblyAI API Key. It can be set as environment variable: `ASSEMBLYAI_API_KEY`.

## Text-to-speech

```python
from livekit.agents import AgentSession
from livekit.plugins import assemblyai

session = AgentSession(
    stt=assemblyai.STT(model="universal-3-6-pro"),
    tts=assemblyai.TTS(voice="jane"),
    # ...
)
```

`voice` is one of the [preset voices](https://www.assemblyai.com/docs/tts/voices). Every voice
speaks exactly one language, and the plugin derives `language` from the voice, so you only pass
one of them:

| Language   | Voices                                                                              |
| ---------- | ----------------------------------------------------------------------------------- |
| English    | `alba`, `anna`, `charles`, `eve`, `george`, `jane` (default), `jean`, `mary`, `michael`, `paul`, `vera` |
| Spanish    | `lola`                                                                              |
| German     | `juergen`                                                                           |
| Italian    | `giovanni`                                                                          |
| Portuguese | `rafael`                                                                            |
| French     | `estelle`                                                                           |

`assemblyai.TTS(language="es")` picks that language's default voice. Voice names are lowercase and
case-sensitive.

Other options:

- `sample_rate`: 8000, 16000, 22050, 24000 (default, the service's native rate), 44100 or 48000.
- `word_timestamps`: per-word timings published as an aligned transcript (default `True`).
- `base_url`: `wss://streaming-tts.us.assemblyai.com` or `wss://streaming-tts.eu.assemblyai.com`
  keeps text and audio in one data zone; the default global endpoint routes to the nearest region.

The service has no speed, pitch or SSML controls, and markup is read aloud as text. Numbers,
currency, dates and common abbreviations are expanded to their spoken form automatically.

### How the plugin uses the API

- Each sentence of the LLM output is sent as its own request (`Generate` + `Flush`), so audio
  starts after the first sentence instead of the whole reply.
- Sessions are pooled and reused across turns: opening a session costs a handshake and counts
  against the per-minute session limit. Call `tts.prewarm()` to open one before the first reply.
- On interruption the plugin sends `Cancel` and keeps the session, dropping any audio that was
  already in flight for the interrupted reply.
- Credential (`1008`) and request (`3006`, including an unsupported voice) errors are not
  retried; transient errors are retried on a new session.
