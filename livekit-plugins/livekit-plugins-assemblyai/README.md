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

`voice` is a voice name from the [voice library](https://www.assemblyai.com/docs/tts/voices), and
`language` a language code (`"es"`, `"es-MX"`) or English name (`"spanish"`). Both are passed to
the server as given, and the API doesn't infer one from the other, so pass the voice's language
along with it:

```python
assemblyai.TTS(voice="lola", language="es")
```

Leave `voice` out to get the server's default voice for the language (`jane` in English). The
server's error for a voice it doesn't speak in that language lists the voices it does.

A custom (cloned) voice is passed as `voice_id` instead of `voice`. Use the `id` that AssemblyAI
returned when the voice was created, or your slug for it. The voice speaks the language of its
reference clip, so `language` can be left out. It must be paired with the session's model (the
language's default, or the one you name with `model_id`), and it can only be used on a host in
the data zone it was created in:

```python
assemblyai.TTS(
    voice_id="01a0da61-138c-77c7-ab98-421431ed74e8",
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
