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

Options map to the Scribe API fields of the same name. For what each feature does, and its availability, pricing and enablement, see the [Scribe docs](https://developers.zoom.us/docs/ai-services/scribe/) and [zoom.ai](https://www.zoom.ai/).

| Option | Default | Description |
|---|---|---|
| `mode` | `"live"` | `"live"` (streaming) or `"fast"` (one request per utterance) |
| `api_key` | `$ZOOM_SCRIBE_API_KEY` | Zoom AI Services API key (a JWT also works). It's sent on every request and reconnect, so it must stay valid while the STT is in use. |
| `silence_duration_ms` | `350` | Live: silence (ms) that ends a turn in Scribe's VAD. 350 is tuned for voice agents; lower is faster, higher keeps sentences with longer pauses together (for transcription, try 600). `None` uses Scribe's default; see the [Scribe docs](https://developers.zoom.us/docs/ai-services/scribe/). |
| `language` | `"en-US"` | A Scribe locale from the [supported languages](https://developers.zoom.us/docs/ai-services/scribe/#supported-languages); base codes such as `"en"` map to their locale. A language this version of the plugin doesn't support raises `ValueError`. |
| `model` | `None` | Scribe speech recognition model (`config.model`). `None` uses Scribe's default model. |
| `language_detection` | `None` | Scribe language detection mode (`config.language_detection.mode`). `None` turns it off. A detected language Scribe reports is in `SpeechData.language`; when Scribe reports none, it is empty. |
| `diarization` | `False` | Label each final with a speaker. With one speaker in the utterance, `SpeechData.speaker_id` is Scribe's label; with several, `speaker_id` is empty and `metadata["speaker_segments"]` has each one's span. In Fast mode labels are per request, so they aren't consistent across utterances. LiveKit's `MultiSpeakerAdapter` needs Live mode. |
| `word_time_offsets` | `False` | Fast: word timings (`config.word_time_offsets`), in `SpeechData.words` |
| `vocabulary` | `None` | Words and phrases that customize recognition (`config.vocabulary.phrases`) |
| `profanity_filter` | `False` | Scribe's profanity filter (`config.profanity_filter`) |
| `text_polishing` | `False` | Scribe's text polishing (`outputs.text_polishing`). The polished text is the transcript, and the recognized text is in `metadata["transcript"]`; if polishing fails, the recognized text is used. Word timings aren't returned with polished text. |
| `base_url` | `https://api.zoom.us/v2/aiservices/scribe` | Scribe API base; Live uses `<base>/live`, Fast `<base>/transcribe`. `live_url` / `fast_url` override either one. A plaintext (`http`/`ws`) URL to a host other than localhost logs a warning. |
| `http_session` | the job's session | `aiohttp.ClientSession` to use |

`stt.update_options(language=..., silence_duration_ms=...)` changes either; open Live streams reconnect with the new settings. An utterance in progress at that moment gets no final transcript; its turn is closed with `END_OF_SPEECH`.

Live returns final transcripts only (no interim results), followed by `END_OF_SPEECH`.
