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

`voice` is one of the [voices](https://www.assemblyai.com/docs/tts/voices). Each voice speaks one
language today, and the plugin derives `language` from the voice, so you only pass one of them:

| Language   | Voices                                                                                                  |
| ---------- | ------------------------------------------------------------------------------------------------------- |
| English    | `alba`, `anna`, `charles`, `eve`, `george`, `iris`, `jane` (default), `jean`, `mary`, `michael`, `paul`, `reid`, `vera` |
| Spanish    | `lola`                                                                                                  |
| German     | `juergen`                                                                                               |
| Italian    | `giovanni`                                                                                              |
| Portuguese | `rafael`                                                                                                |
| French     | `estelle`                                                                                               |

`assemblyai.TTS(language="es")` (or `"spanish"`) uses the server's default voice for that language.
A voice or a voice/language pairing the table doesn't list is passed through for the server to
judge, and the server's error lists the voices available.

A custom (cloned) voice is passed as `voice_id` instead of `voice`. Use the `id` that AssemblyAI
returned when the voice was created, or your slug for it. The voice speaks the language of its
reference clip, so `language` can be left out. It must be paired with the session's model, and it
can only be used on a host in the data zone it was created in:

```python
assemblyai.TTS(
    voice_id="01a0da61-138c-77c7-ab98-421431ed74e8",
    model_id="assemblyai-tts-mul-4brl-2026-10-12",  # optional: a dated model name or an alias
    base_url="wss://streaming-tts.us.assemblyai.com",
)
```

Other options:

- `sample_rate`: 8000, 16000, 22050, 24000 (default, the service's native rate), 44100 or 48000.
- `word_timestamps`: per-word timings published as an aligned transcript (default `True`).
- `base_url`: `wss://streaming-tts.us.assemblyai.com` or `wss://streaming-tts.eu.assemblyai.com`
  keeps text and audio in one data zone; the default global endpoint routes to the nearest region.

The service has no speed, pitch or SSML controls. Numbers, currency, dates and common
abbreviations are expanded to their spoken form automatically; keep LLM markup out of the speech
with `AgentSession(tts_text_transforms=["filter_markdown", "filter_emoji"])`.

### How the plugin uses the API

- Each sentence of the LLM output is sent as its own request (`Generate` + `Flush`), so audio
  starts after the first sentence instead of the whole reply. `synthesize()` runs over the same
  pooled sessions.
- Sessions are pooled and reused across turns: opening a session costs a handshake and counts
  against the per-minute session limit. Call `tts.prewarm()` to open one before the first reply.
- On interruption the plugin sends `Cancel` and keeps the session, dropping any audio that was
  already in flight for the interrupted reply.
- Errors are retried on a new session when the server marks them retryable (credential and
  request errors, such as an unsupported voice or a custom voice that is missing or not paired
  with the model, are not). A new session waits out any
  `retry_after_seconds` the server asked for; a wait longer than the connect timeout ends the
  retries instead, so a `FallbackAdapter` can move on.
