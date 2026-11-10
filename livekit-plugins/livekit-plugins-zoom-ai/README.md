# Zoom AI plugin for LiveKit Agents

Support for [Zoom Voice Agent](https://developers.zoom.us/docs/ai-services/voice-agent/) speech-to-speech (see [Voice Agent](#voice-agent-realtime) below) and [Zoom Scribe](https://developers.zoom.us/docs/ai-services/scribe/) speech-to-text, in two modes:

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

The plugin needs a Zoom AI Services API key. Sign up at [zoom.ai](https://www.zoom.ai/) and create an API key with the Scribe or Voice Agent permissions. Plans and pricing are there too. Existing accounts can also create keys in [Zoom Platform Studio](https://platform.zoom.us/) under Credentials → API Keys.

Set it in your `.env` file:

```
ZOOM_AI_API_KEY=<your_zoom_api_key>
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
| `api_key` | `$ZOOM_AI_API_KEY` | Zoom AI Services API key |
| `language` | `"en-US"` | Recognition language ([supported languages](https://developers.zoom.us/docs/ai-services/scribe/speech-understanding/)); a regional variant Scribe doesn't list uses its base language's locale (`en-GB` → `en-US`) |
| `model` | `None` | Recognition model; `None` uses Scribe's default |
| `language_detection` | `None` | Language detection mode (Live supports `"continuous"`); the language Scribe detects is in `SpeechData.language`. With `zoom-scribe-multilingual` in Live, set it to get each final's language |
| `silence_duration_ms` | `350` | Live: silence (ms) that ends a turn; `None` uses Scribe's default |
| `diarization` | `False` | Speaker labels: `SpeechData.speaker_id`, or `metadata["speaker_segments"]` when several people speak in one final |
| `word_time_offsets` | `False` | Fast: word timings in `SpeechData.words` |
| `channel_separation` | `False` | Fast: recognize each stereo channel separately; channels in `metadata["channel_segments"]` |
| `vocabulary` | `None` | Words and phrases to recognize, such as names and product terms |
| `text_polishing` | `False` | Polished text as the transcript, with the original in `metadata["transcript"]`; if polishing fails, the original text. `en-US` only |
| `extra_kwargs` | `None` | Other Scribe request fields, e.g. `{"config": {"language_detection": {"candidate_languages": ["en-US", "fr-FR"]}}}` to restrict language detection |
| `base_url` | `https://api.zoom.us/v2/aiservices/scribe` | Scribe API base URL |

## Voice Agent (realtime)

`zoom_ai.realtime.RealtimeModel` runs a conversation on the [Voice Agent API](https://developers.zoom.us/docs/ai-services/voice-agent/live-mode/). One WebSocket carries the caller's audio in and the agent's speech, transcripts and function calls back; Zoom runs recognition, the model and synthesis. Zoom detects turns and handles barge-in, so no VAD, STT or TTS is needed:

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import zoom_ai

session = AgentSession(llm=zoom_ai.realtime.RealtimeModel(voice="jenna"))
await session.start(Agent(instructions="You are a friendly support agent."), room=ctx.room)
```

A runnable version with a function tool is [`examples/voice_agents/zoom_voice_agent.py`](../../examples/voice_agents/zoom_voice_agent.py).

It can also be the model of a single agent in a multi-agent handoff (`Agent(llm=zoom_ai.realtime.RealtimeModel())`); see [`examples/voice_agents/zoom_multi_agent.py`](../../examples/voice_agents/zoom_multi_agent.py).

Function tools work as with any model: Zoom asks for the call, the agent runs it, and the plugin returns the result and asks for the follow-up reply.

To use your own OpenAI or Anthropic account for the conversation model, pass its key in `provider_headers`:

```python
zoom_ai.realtime.RealtimeModel(
    llm_provider="openai",
    model="<provider-model-id>",
    provider_headers={"X-Api-Key": "<your-openai-key>"},
    provider_metadata={"context_window": 1000000, "max_output_tokens": 128000},
)
```

### Voice Agent options

| Option | Default | Description |
|---|---|---|
| `voice` | `"jenna"` | `alex`, `ariel`, `ava`, `brooke`, `hazel`, `jenna`, `mark`, `reid`, `scott` (American English) or `tony` (British English) |
| `model` | `"gemma-4"` | Conversational model; with your own provider, one of its model IDs |
| `llm_provider` | `"zoom"` | `"zoom"`, `"openai"` or `"anthropic"` |
| `provider_headers` | `None` | Your provider's API key headers; required unless `llm_provider="zoom"` |
| `provider_metadata` | `None` | Your model's `context_window` and `max_output_tokens` |
| `vocabulary` | `None` | Names and terms that improve transcription accuracy |
| `reasoning_effort` | `None` | Reasoning level, such as `"low"` |
| `response_token_limit` | `None` | Maximum tokens per response; `None` uses Zoom's default (`"inf"`) |
| `tool_choice` | `None` | `"auto"`, `"none"` or `"required"`; `None` uses Zoom's default (`"auto"`) |
| `api_key` | `$ZOOM_AI_API_KEY` | Zoom AI Services API key |
| `url` | `wss://api.zoom.us/v2/ws/aiservices/voice-agent/live` | Voice Agent endpoint |

`voice`, `tool_choice` and `vocabulary` can be changed on a live session with `update_options()`; instructions and tools follow the agent.

### Debugging

The session emits every Zoom server event, unwrapped from its envelope, as `zoom_server_event_received`. To see the raw protocol, listen on the realtime session directly, for example `rt_session.on("zoom_server_event_received", print)` on a session from `model.session()`. Plugin logs are under the `livekit.plugins.zoom_ai` logger.

### Voice Agent limitations

These come from the Voice Agent API:

- English only.
- Replies can't take their own instructions: `generate_reply(instructions=...)` is ignored with a warning, and the reply follows the session instructions. Put greetings and the like in the agent's `instructions`.
- Messages can't be added to a running conversation; only function results are sent. A chat history that exists before the session starts (an agent handoff) is added to the instructions instead.
- Turn detection, barge-in, noise reduction and history trimming are Zoom's and can't be configured or turned off.
- A session lasts up to 60 minutes. When Zoom ends it, the plugin reconnects and carries the conversation over in the instructions.
- Audio only: video frames are dropped.
