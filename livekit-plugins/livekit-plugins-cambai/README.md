# Camb.ai Plugin for LiveKit Agents

Text-to-Speech and realtime speech-to-speech translation for [Camb.ai](https://camb.ai), powered by MARS technology.

## Features

- High-quality neural text-to-speech with MARS series models
- Multiple model variants (mars-flash, mars-pro)
- Enhanced pronunciation for names and places
- Support for 140+ languages
- Real-time HTTP streaming
- Pre-built voice library
- Realtime speech-to-speech translation: speech in one language, speech in another, in the speaker's voice

## Installation

```bash
pip install livekit-plugins-cambai
```

## Prerequisites

You'll need a Camb.ai API key. Set it as an environment variable:

```bash
export CAMB_API_KEY=your_api_key_here
```

Or obtain it from [Camb.ai Studio](https://studio.camb.ai/public/onboarding).

## Quick Start

```python
import asyncio
from livekit.plugins.cambai import TTS

async def main():
    # Initialize TTS (uses CAMB_API_KEY env var)
    tts = TTS()

    # Synthesize speech
    stream = tts.synthesize("Hello from Camb.ai!")
    audio_frame = await stream.collect()

    # Save to file
    with open("output.wav", "wb") as f:
        f.write(audio_frame.to_wav_bytes())

asyncio.run(main())
```

## List Available Voices

```python
import asyncio
from livekit.plugins.cambai import list_voices

async def main():
    voices = await list_voices()
    for voice in voices:
        print(f"{voice['name']} ({voice['id']}): {voice['gender']}, {voice['language']}")

asyncio.run(main())
```

## Select a Specific Voice

```python
tts = TTS(voice_id=147320)
stream = tts.synthesize("Using a specific voice!")
```

## Model Selection

Camb.ai offers multiple MARS models for different use cases:

```python
# Faster inference, 22050 Hz (default)
tts = TTS(model="mars-flash")

# Higher quality, 48000 Hz
tts = TTS(model="mars-pro")
```

## Advanced Configuration

```python
tts = TTS(
    api_key="your-api-key",  # Or use CAMB_API_KEY env var
    voice_id=147320,  # Voice ID from list-voices
    language="en-us",  # BCP-47 locale
    model="mars-pro",  # MARS model variant
    output_format="pcm_s16le",  # Audio format
    enhance_named_entities=True,  # Better pronunciation for names/places
)
```

## Usage with LiveKit Agents

```python
from livekit import agents
from livekit.plugins.cambai import TTS

async def entrypoint(ctx: agents.JobContext):
    # Connect to room
    await ctx.connect()

    # Initialize TTS
    tts = TTS(language="en-us")

    # Synthesize and publish
    stream = tts.synthesize("Hello from LiveKit with Camb.ai!")
    audio_frame = await stream.collect()

    # Publish to room
    source = agents.AudioSource(tts.sample_rate, tts.num_channels)
    track = agents.LocalAudioTrack.create_audio_track("tts", source)
    await ctx.room.local_participant.publish_track(track)
    await source.capture_frame(audio_frame)
```

## Configuration Options

### TTS Constructor Parameters

- **api_key** (str | None): Camb.ai API key
- **voice_id** (int): Voice ID to use (default: 147320)
- **language** (str): BCP-47 locale (default: "en-us")
- **model** (SpeechModel): MARS model variant (default: "mars-flash")
- **output_format** (OutputFormat): Audio format (default: "pcm_s16le")
- **enhance_named_entities** (bool): Enhanced pronunciation (default: False)
- **sample_rate** (int | None): Audio sample rate (auto-detected from model if None)
- **base_url** (str): API base URL
- **http_session** (httpx.AsyncClient | None): Reusable HTTP session

### Available Models

- **mars-flash**: Faster inference, 22050 Hz (default)
- **mars-pro**: Higher quality synthesis, 48000 Hz

### Output Formats

- **pcm_s16le**: 16-bit PCM (recommended for streaming)
- **pcm_s32le**: 32-bit PCM (highest quality)
- **wav**: WAV with headers
- **flac**: Lossless compression
- **adts**: ADTS streaming format

## API Reference

### TTS Class

Main text-to-speech interface.

**Methods:**
- `synthesize(text: str) -> ChunkedStream`: Synthesize text to speech
- `update_options(**kwargs)`: Update voice settings dynamically
- `aclose()`: Clean up resources

**Properties:**
- `model` (str): Current MARS model name
- `provider` (str): Provider name ("Camb.ai")
- `sample_rate` (int): Audio sample rate (22050 or 48000 Hz depending on model)
- `num_channels` (int): Number of audio channels (1)

### list_voices Function

```python
async def list_voices(
    api_key: str | None = None,
    base_url: str = "https://client.camb.ai/apis",
) -> list[dict]
```

Returns list of voice dicts with: id, name, gender, age, language.

## Multi-Language Support

Camb.ai supports 140+ languages. Specify using BCP-47 locales:

```python
# French
tts = TTS(language="fr-fr", voice_id=...)

# Spanish
tts = TTS(language="es-es", voice_id=...)

# Japanese
tts = TTS(language="ja-jp", voice_id=...)
```

## Dynamic Options

Update TTS settings without recreating the instance:

```python
tts = TTS()

# Change voice
tts.update_options(voice_id=12345)

# Change model
tts.update_options(model="mars-pro")
```

## Error Handling

The plugin handles errors according to LiveKit conventions:

```python
from livekit.agents import APIStatusError, APIConnectionError, APITimeoutError

try:
    stream = tts.synthesize("Hello!")
    audio = await stream.collect()
except APIStatusError as e:
    print(f"API error: {e.status_code} - {e.message}")
except APIConnectionError as e:
    print(f"Connection error: {e}")
except APITimeoutError as e:
    print(f"Request timed out: {e}")
```

## Future Features

Coming soon:
- GCP Vertex AI integration
- Voice cloning via custom voice creation
- Voice generation from text descriptions
- WebSocket streaming for real-time applications

## Links

- [Camb.ai Documentation](https://docs.camb.ai/)
- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [GitHub Repository](https://github.com/livekit/agents)

## Realtime speech-to-speech translation

`cambai.experimental.realtime.RealtimeModel` translates speech to speech: the participant speaks one
language and the model returns the same utterance spoken in another, along with a
transcript of what was said and the translated text. It replaces the usual
STT + LLM + TTS chain with a single connection.

Drop it into an `AgentSession` like any other realtime model:

```python
from livekit.agents import AgentSession
from livekit.plugins import cambai

session = AgentSession(
    llm=cambai.experimental.realtime.RealtimeModel(
        source_language="en-US",
        target_language="fr-FR",
    ),
)
```

No VAD is needed: the endpoint segments utterances itself, so the model reports
server-side turn detection and the session does not run its own barge-in detection. That
matters for translation, where the speaker never stops talking and would otherwise be
treated as interrupting the agent.

To publish a translated track per speaker instead, drive the session directly — see
`examples/other/translation/camb_realtime_translator.py`:

```python
from livekit import rtc
from livekit.plugins import cambai

model = cambai.experimental.realtime.RealtimeModel(
    source_language="en-US",   # what the speaker says
    target_language="fr-FR",   # what the room hears
    mode="fast",
)
session = model.session()

translated = rtc.AudioSource(24000, 1)
await ctx.room.local_participant.publish_track(
    rtc.LocalAudioTrack.create_audio_track("translated-fr-FR", translated),
    rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
)


async def forward(track: rtc.Track) -> None:
    async for ev in rtc.AudioStream(track):
        session.push_audio(ev.frame)


@session.on("input_audio_transcription_completed")
def _on_transcript(ev) -> None:
    print("source:", ev.transcript)


@session.on("generation_created")
def _on_generation(ev) -> None:
    async def play() -> None:
        async for msg in ev.message_stream:
            async for frame in msg.audio_stream:
                await translated.capture_frame(frame)

    asyncio.create_task(play())
```

Each generation also carries `msg.text_stream`, the translated text, which pairs with the
source transcript above for captions.

Audio is 24 kHz mono PCM16 in both directions; frames at any other rate are resampled for
you. `voice_id` synthesizes the translation with one of your cloned voices instead of a
built-in one, and `base_url` points the session at a non-production deployment.

### Choosing a mode

`mode="fast"` starts speaking sooner; `mode="slow"` covers a longer language list. Both
translate every complete utterance they are given — measured against `realtime.camb.ai`
on English recordings from 3.9s to 12s, neither mode dropped a finished sentence, and
translation quality was comparable in both.

What both modes ignore is an *incomplete* utterance. Feeding audio that stops mid-sentence
leaves that fragment untranslated, which is correct but surprising if you are replaying a
file you cut at an arbitrary offset: cut on pauses, or accept that the trailing fragment
goes nowhere. A live microphone raises this only at the very end of a call.

### Turn taking

The endpoint segments utterances itself and streams translations continuously; it emits no
speech-start or speech-stop events, so `capabilities.turn_detection` is `False`. Nothing
needs committing and no reply needs requesting — `commit_audio`, `clear_audio` and
`interrupt` are inert, and `generate_reply` hands back the translation the next utterance
produces.

Note that a conversational orchestrator is a poor fit for a translator: the speaker never
stops talking, so anything that treats incoming speech as an interruption will cancel the
translation mid-playback. Drive the session directly, as above.

## License

Apache License 2.0
