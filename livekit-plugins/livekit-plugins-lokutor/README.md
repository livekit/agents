# LiveKit Plugins Lokutor

**Lokutor** speech-to-text and text-to-speech for [LiveKit Agents](https://docs.livekit.io/agents/).

Lokutor builds its own small speech models: Conv 2.0 for recognition and Versa 2.0 for synthesis, each one model for nine languages (English, Spanish, Catalan, Galician, Basque, Portuguese, French, Italian and German). See [lokutor.com](https://lokutor.com) and the [API docs](https://docs.lokutor.com).

## Installation

```bash
pip install livekit-plugins-lokutor
```

Set `LOKUTOR_API_KEY`, or pass `api_key=` to `STT` and `TTS`.

## Usage

### In an agent

```python
from livekit.agents import AgentSession
from livekit.plugins import lokutor, openai, silero

session = AgentSession(
    stt=lokutor.STT(language="es"),
    llm=openai.LLM(),
    tts=lokutor.TTS(voice="F1", language="es"),
    vad=silero.VAD.load(),
)
```

### Speech-to-text

`STT` streams over `WS /ws/stt`: interim transcripts while the user speaks, then a final transcript when the server sees the utterance end (a pause of about 0.7 s). `recognize()` transcribes one clip with `POST /stt/transcribe`. The text comes back written out, with punctuation, casing, digits and e-mail addresses.

```python
stt = lokutor.STT(language="en")
event = await stt.recognize(frames)  # a list of rtc.AudioFrame, any sample rate
print(event.alternatives[0].text)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | env `LOKUTOR_API_KEY` | Lokutor API key |
| `language` | `str` | detected | One of the nine languages; regional codes such as `es-ES` are accepted. Set it whenever you know it: it keeps one-word replies reliable and sets how the text is written. Any other value is left to detection |
| `base_url` | `str` | `"wss://api.lokutor.com"` | API base URL |

### Text-to-speech

```python
tts = lokutor.TTS(voice="F1", language="en")
async with tts:
    stream = tts.stream()
    stream.push_text("Hello, world!")
    stream.end_input()
    async for audio in stream:
        print(f"Got audio: {audio.frame.duration:.2f}s")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | env `LOKUTOR_API_KEY` | Lokutor API key |
| `voice` | `str` | `"F1"` | Voice slot, `F1`-`F5` (female) or `M1`-`M5` (male). Each language has its own native speaker in each slot |
| `language` | `str \| None` | `"en"` | One of the nine languages; any other value uses the English speakers |
| `visemes` | `bool` | `False` | Lip-sync viseme data |
| `speed`, `steps` | | not sent | Accepted for compatibility; `versa-2.0` sets its own pace and step schedule, so they do not change the audio |
| `base_url` | `str` | `"wss://api.lokutor.com"` | API base URL |

Audio comes out at 44.1 kHz mono; the pipeline resamples if it needs another rate.

## License

Apache 2.0
