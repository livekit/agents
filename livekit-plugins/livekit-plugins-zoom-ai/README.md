# Zoom AI plugin for LiveKit Agents

Support for [Zoom Scribe](https://developers.zoom.us/docs/ai-services/scribe/) speech-to-text, in two modes:

- **Live** ([docs](https://developers.zoom.us/docs/ai-services/scribe/live-mode/)): streaming over WebSocket. Scribe's own VAD ends each speech turn and returns its final transcript.
- **Fast** ([docs](https://developers.zoom.us/docs/ai-services/scribe/fast-mode/)): one HTTP request per utterance. Not streaming, so `AgentSession` pairs it with a VAD.

See [zoom.ai](https://www.zoom.ai/) for all Zoom AI Services APIs and features.

## Installation

```bash
pip install livekit-plugins-zoom-ai
```

Or install with the LiveKit Agents extra:

```bash
uv add "livekit-agents[zoom-ai]"
```

## Pre-requisites

The plugin needs a Zoom AI Services API key (or JWT). Create one in [Zoom Platform Studio](https://platform.zoom.us/) under Credentials → API Keys, with the Scribe permissions.

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

### Without an agent

Both modes also work on their own, for example for transcription. Outside an agent job, pass an `aiohttp` session:

```python
stt = zoom_ai.STT(http_session=session, silence_duration_ms=600)  # Live
stream = stt.stream()  # push_frame() audio, then iterate the stream for SpeechEvents

event = await zoom_ai.STT(mode="fast", http_session=session).recognize(frames)  # Fast
```

For transcription, a longer `silence_duration_ms` (such as 600) keeps sentences with pauses together and can improve accuracy; the 350 ms default is tuned for voice agents.

## Options

Options map to the Scribe API fields of the same name; see the [Scribe docs](https://developers.zoom.us/docs/ai-services/scribe/) for what each one does and its availability. For a Scribe feature without an option here, pass its fields in `extra_kwargs`.

| Option | Default | Description |
|---|---|---|
| `mode` | `"live"` | `"live"` (streaming) or `"fast"` (one request per utterance) |
| `api_key` | `$ZOOM_SCRIBE_API_KEY` | Zoom AI Services API key or JWT |
| `language` | `"en-US"` | Recognition language ([supported languages](https://developers.zoom.us/docs/ai-services/scribe/#supported-languages)) |
| `model` | `None` | Recognition model; `None` uses Scribe's default |
| `language_detection` | `None` | Language detection mode; the detected language is in `SpeechData.language` |
| `silence_duration_ms` | `350` | Live: silence (ms) that ends a turn; `None` uses Scribe's default |
| `diarization` | `False` | Speaker labels: `SpeechData.speaker_id`, or `metadata["speaker_segments"]` when several people speak in one final |
| `word_time_offsets` | `False` | Fast: word timings in `SpeechData.words` |
| `vocabulary` | `None` | Words and phrases to recognize, such as names and product terms |
| `text_polishing` | `False` | Polished text as the transcript, with the original in `metadata["transcript"]`; if polishing fails, the original text |
| `extra_kwargs` | `None` | Other Scribe request fields, e.g. `{"config": {"profanity_filter": True}}` |
| `base_url` | `https://api.zoom.us/v2/aiservices/scribe` | Scribe API base URL |
