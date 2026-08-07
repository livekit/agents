# Baseten plugin for LiveKit Agents

Support for [Baseten](https://baseten.co/)-hosted models in LiveKit Agents, including **STT** (Speech-to-Text), **TTS** (Text-to-Speech), and **LLM** (Large Language Model) integrations.

## Installation

```bash
pip install livekit-plugins-baseten
```

## Pre-requisites

You'll need an API key from Baseten. It can be set as an environment variable: `BASETEN_API_KEY`

You also need to deploy a model to Baseten and will need your model endpoint to configure the plugin.

## STT (Speech-to-Text)

The STT plugin connects to Baseten's [Whisper Streaming](https://docs.baseten.co/reference/inference-api/predict-endpoints/streaming-transcription-api) WebSocket endpoint for real-time transcription. It works with both **truss** and **chain** deployments.

### Recommended model

[Whisper v3 Turbo – WebSocket](https://www.baseten.co/library/whisper-streaming-large-v3/)

### Endpoint URL formats

| Deployment type | URL pattern |
|---|---|
| **Truss** | `wss://model-{model_id}.api.baseten.co/environments/production/websocket` |
| **Chain** | `wss://chain-{chain_id}.api.baseten.co/environments/production/websocket` |

### Basic usage

You can specify the endpoint in three ways:

```python
from livekit.plugins import baseten

# 1. Using a truss model ID (recommended for truss deployments)
stt = baseten.STT(
    api_key="your-baseten-api-key",  # or set BASETEN_API_KEY env var
    model_id="your-model-id",
    language="en",
)

# 2. Using a chain ID (recommended for chain deployments)
stt = baseten.STT(
    api_key="your-baseten-api-key",
    chain_id="your-chain-id",
    language="en",
)

# 3. Using a full endpoint URL (for custom routing or deployment URLs)
stt = baseten.STT(
    api_key="your-baseten-api-key",
    model_endpoint="wss://model-{model_id}.api.baseten.co/environments/production/websocket",
    language="en",
)
```

### Configuration options

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `BASETEN_API_KEY` env var | Baseten API key |
| `model_endpoint` | `BASETEN_MODEL_ENDPOINT` env var | Full WebSocket URL (takes priority over `model_id`/`chain_id`) |
| `model_id` | — | Baseten truss model ID; auto-constructs the endpoint URL |
| `chain_id` | — | Baseten chain ID; auto-constructs the endpoint URL |
| `language` | `"en"` | BCP-47 language code (use `"auto"` for auto-detection) |
| `language_options` | `[]` | Restrict auto-detection to these codes, e.g. `["en", "de"]`. More reliable than `"auto"` on short telephony utterances. Requires Whisper runtime v0.5.0+ |
| `encoding` | `"pcm_s16le"` | Audio encoding (`pcm_s16le` or `pcm_mulaw`) |
| `sample_rate` | `16000` | Audio sample rate in Hz |
| `enable_partial_transcripts` | `True` | Emit interim transcripts while the speaker is talking |
| `partial_transcript_interval_s` | `1.0` | Interval (seconds) between partial transcript updates |
| `final_transcript_max_duration_s` | `30` | Max seconds of audio before forcing a final transcript |
| `show_word_timestamps` | `True` | Include word-level timestamps in results |
| `vad_threshold` | `0.5` | Server-side VAD speech probability threshold (0.0–1.0) |
| `vad_min_silence_duration_ms` | `300` | Minimum silence (ms) to mark end of speech |
| `vad_speech_pad_ms` | `30` | Padding (ms) added around detected speech |

### Full voice pipeline example

```python
import os
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, inference
from livekit.plugins import baseten, openai, noise_cancellation
from livekit.agents.inference import TurnDetector

BASETEN_API_KEY = os.getenv("BASETEN_API_KEY")
whisper_model_id = "your-whisper-model-id"  # or use chain_id for chain deployments
orpheus_model_id = "your-orpheus-model-id"


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=baseten.STT(
            api_key=BASETEN_API_KEY,
            model_id=whisper_model_id,  # or chain_id="your-chain-id"
            language="en",
            enable_partial_transcripts=True,
        ),
        llm=openai.LLM(
            api_key=BASETEN_API_KEY,
            base_url="https://inference.baseten.co/v1",
            model="openai/gpt-oss-120b",
        ),
        tts=baseten.TTS(
            api_key=BASETEN_API_KEY,
            model_endpoint=(
                f"https://model-{orpheus_model_id}"
                ".api.baseten.co/environments/production/predict"
            ),
        ),
        vad=inference.VAD(),
        turn_detection=TurnDetector(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
```

## TTS (Text-to-Speech)

The TTS plugin calls Baseten-hosted TTS models (e.g. [Orpheus 3B](https://www.baseten.co/library/orpheus-tts/)) over HTTP.

```python
tts = baseten.TTS(
    api_key="your-baseten-api-key",
    model_endpoint="https://model-{model_id}.api.baseten.co/environments/production/predict",
    voice="tara",
    language="en",
)
```

## Qwen3 (STT + TTS)

Baseten also hosts **Qwen3-ASR** and **Qwen3-TTS**. They speak different wire
protocols from the Whisper/Orpheus trusses, so select them with `model` — the
same `baseten.STT` / `baseten.TTS` classes handle both.

```python
from livekit.agents import AgentSession
from livekit.plugins import baseten

session = AgentSession(
    stt=baseten.STT(model="qwen3-asr", model_id="your-qwen3-asr-model-id"),
    tts=baseten.TTS(model="qwen3-tts", model_id="your-qwen3-tts-model-id",
                    voice="your-voice"),
    # llm=...
)
```

| | `model="whisper"` / `"orpheus"` (default) | `model="qwen3-asr"` / `"qwen3-tts"` |
|---|---|---|
| STT audio | raw binary PCM | base64 `input_audio_buffer.append` |
| STT results | `message_type` / `transcript` | `type: "transcription"` / `segments[].text` |
| TTS transport | HTTP, or WS with an `__END__` sentinel | `session.config` → `input.text` → `input.done` |
| TTS voices | preset names (`tara`) | registered voice clones |

Several defaults follow the selected model, matching each deployment's own:

| Parameter | `whisper` / `orpheus` | `qwen3-asr` / `qwen3-tts` |
|---|---|---|
| `language` | `en` | `auto` (STT) / `Auto` (TTS) |
| `partial_transcript_interval_s` | `1.0` | `0.5` |
| `vad_min_silence_duration_ms` | `300` | `500` |
| `vad_speech_pad_ms` | `30` | `100` |
| `show_word_timestamps` | on | off (needs `STREAM_ALIGNER=mms` on the deployment) |
| `voice` | `tara` | required — a registered clone |

### Qwen3 TTS voices

Qwen3-TTS Base ships **no built-in speakers** — there is no `tara` equivalent.
Register a clone from 10–20s of clean speech, then pass its name as `voice`:

```python
from livekit.plugins.baseten import list_voices, register_voice

await register_voice(
    model_endpoint="wss://model-{model_id}.api.baseten.co/environments/production/websocket",
    name="my_voice",
    ref_audio_path="./reference.wav",
    ref_text="Transcript of the reference audio.",
)
await list_voices(model_endpoint=...)  # {"voices": [...], "uploaded_voices": [...]}
```

The server stores uploaded voices on the container's local disk, so a voice
registered at runtime lives on **one replica** and is lost when that container
restarts. For anything beyond single-replica testing, bake the reference audio
into the deployment (`REQUIRED_VOICES`) so every replica starts with it, or pass
`ref_audio`/`ref_text` to clone inline on each session.

### Qwen3-only options

`task_type` (`Base` cloning, or `CustomVoice`/`VoiceDesign` deployments),
`instructions`, `max_new_tokens`, `initial_codec_chunk_frames`,
`x_vector_only_mode`, `ref_audio`/`ref_text`, and `extra_config` (merged into
`session.config` for server fields newer than this plugin) apply to
`model="qwen3-tts"` only.

## LLM (Large Language Model)

The LLM plugin wraps Baseten's OpenAI-compatible inference endpoint.

```python
llm = baseten.LLM(
    api_key="your-baseten-api-key",
    model="openai/gpt-oss-120b",
)
```

## Documentation

- [LiveKit STT integration guide](https://docs.livekit.io/agents/integrations/stt/baseten/)
- [LiveKit TTS integration guide](https://docs.livekit.io/agents/integrations/tts/baseten/)
- [Baseten Whisper Streaming docs](https://docs.baseten.co/reference/inference-api/predict-endpoints/streaming-transcription-api)
- [Baseten Model Library](https://www.baseten.co/library/)
