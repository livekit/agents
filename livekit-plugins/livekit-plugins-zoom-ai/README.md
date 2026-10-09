# Zoom AI plugin for LiveKit Agents

Support for [Zoom Scribe](https://developers.zoom.us/docs/ai-services/scribe/) speech-to-text, in two modes:

- **Live** ([docs](https://developers.zoom.us/docs/ai-services/scribe/live-mode/)): streaming over WebSocket. Scribe's own VAD ends each speech turn and returns its final transcript.
- **Fast** ([docs](https://developers.zoom.us/docs/ai-services/scribe/fast-mode/)): one HTTP request per utterance. Not streaming, so `AgentSession` pairs it with a VAD.

> This integration is maintained by [Zoom](https://zoom.com/).

## Installation

```bash
pip install livekit-plugins-zoom-ai
```

Or install with the LiveKit Agents extra:

```bash
uv add "livekit-agents[zoom-ai]"
```

## Pre-requisites

The plugin needs a Zoom AI Services API key or JWT. Create one in [Zoom Platform Studio](https://platform.zoom.us/) under Credentials.

Set it in your `.env` file:

```
ZOOM_SCRIBE_API_KEY=<your_zoom_api_key>
```

## Usage

### Scribe Live (streaming)

Scribe detects the end of each turn, so let the STT drive turn detection:

```python
from livekit.agents import AgentSession
from livekit.plugins import zoom_ai

session = AgentSession(
    stt=zoom_ai.STT(language="en-US"),
    turn_handling={"turn_detection": "stt"},
    # ... llm, tts, etc.
)
```

### Scribe Fast (per utterance)

```python
from livekit.agents import AgentSession
from livekit.plugins import silero, zoom_ai

session = AgentSession(
    stt=zoom_ai.STT(mode="fast"),
    vad=silero.VAD.load(),  # cuts the audio into utterances
    # ... llm, tts, etc.
)
```

Fast also works standalone: `event = await zoom_ai.STT(mode="fast").recognize(frames)`.

## Options

| Option | Default | Description |
|---|---|---|
| `mode` | `"live"` | `"live"` (streaming) or `"fast"` (one request per utterance) |
| `api_key` | `$ZOOM_SCRIBE_API_KEY` | Zoom AI Services API key or JWT |
| `language` | `"en-US"` | `en-US`, `zh-CN`, `ja-JP`, `es-ES`, `it-IT`, `fr-FR`, `de-DE`, `ar-SA`, `ar-AE`, `pt-BR`, `pt-PT`. Base codes such as `"en"` are mapped. |
| `diarization` | `False` | Fast only: label speakers (`SpeechData.speaker_id`) |

`stt.update_options(language=...)` changes the language; open Live streams reconnect with the new setting.

Live returns final transcripts only (no interim results), followed by `END_OF_SPEECH`.
