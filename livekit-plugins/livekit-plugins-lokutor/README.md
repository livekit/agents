# LiveKit Plugins Lokutor

**Lokutor** TTS integration for [LiveKit Agents](https://docs.livekit.io/agents/).

Lokutor is a cost-effective Text-to-Speech API that runs on CPU infrastructure, supporting 10 voices (F1-F5, M1-M5) and 30+ languages. See [lokutor.com](https://lokutor.com) for more details.

## Installation

```bash
pip install livekit-plugins-lokutor
```

## Usage

### In an Agent pipeline

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import lokutor, openai, deepgram, silero

session = AgentSession(
    stt=deepgram.STT(),
    llm=openai.LLM(),
    tts=lokutor.TTS(
        api_key="your-api-key",
        voice="F1",
        language="en",
    ),
    vad=silero.VAD.load(),
)
```

### Standalone TTS

```python
import asyncio
from livekit import rtc
from livekit.plugins import lokutor

async def main():
    tts = lokutor.TTS(api_key="...", voice="F1")
    async with tts:
        stream = tts.stream()
        stream.push_text("Hello, world!")
        stream.end_input()
        async for audio in stream:
            print(f"Got audio: {audio.frame.duration:.2f}s")

asyncio.run(main())
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | env `LOKUTOR_API_KEY` | Lokutor API key |
| `voice` | `str` | `"F1"` | Voice ID (`F1`-`F5`, `M1`-`M5`) |
| `language` | `str \| None` | `"en"` | Language code (`en`, `es`, `fr`, `pt`, `ko`, etc.) |
| `speed` | `float` | `1.05` | Speed multiplier (0.5–2.0) |
| `steps` | `int` | `5` | Diffusion steps: lower = faster, higher = quality (3–10) |
| `visemes` | `bool` | `False` | Enable lip-sync viseme data |
| `sample_rate` | `int` | `44100` | Audio sample rate in Hz |
| `base_url` | `str` | `"wss://api.lokutor.com"` | API base URL |

## Voices

| ID | Gender |
|----|--------|
| F1–F5 | Female |
| M1–M5 | Male |

## Languages

English, Spanish, French, Portuguese, Korean, and 30+ more. Full list at [docs.lokutor.com](https://docs.lokutor.com/voices-languages-models).

## License

Apache 2.0
