# LiveKit Plugins HUMAIN

**HUMAIN Voice** text-to-speech for [LiveKit Agents](https://docs.livekit.io/agents/).

[HUMAIN Voice](https://www.humain.com/voice) is an Arabic-first speech platform with data residency in Saudi Arabia. Its TTS speaks Arabic in Saudi, Levantine, Moroccan and Egyptian voices and English, and handles Arabic-English code-switching within a sentence. See the [API docs](https://docs.voice.humain.com).

LiveKit docs: [text-to-speech](https://docs.livekit.io/agents/models/tts/humain/) (live with the next release).

## Installation

```bash
pip install livekit-plugins-humain
```

Set `HUMAIN_API_KEY`, or pass `api_key=` to `TTS`. Create a key in the [HUMAIN Voice console](https://console.voice.humain.com).

## Usage

### In an agent

```python
from livekit.agents import AgentSession
from livekit.plugins import humain

session = AgentSession(
    stt=...,
    llm=...,
    tts=humain.TTS(voice="mul_nourah"),
    vad=...,
)
```

### Text-to-speech

`TTS` uses HUMAIN's Socket.IO API through the official [`humain-voice`](https://pypi.org/project/humain-voice/) SDK. One connection per `TTS` serves every request. Each request carries one complete text, so `stream()` splits the agent's text into sentences and sends one request per sentence; `synthesize()` sends the whole text in one request. `prewarm()` connects and resolves the voice ahead of the first request.

```python
tts = humain.TTS(voice="mul_sara")
async with tts:
    stream = tts.stream()
    stream.push_text("هلا وغلا فيك. وش أقدر أخدمك فيه اليوم؟")
    stream.end_input()
    async for audio in stream:
        print(f"Got audio: {audio.frame.duration:.2f}s")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voice` | `str` | `"mul_nourah"` | Voice label from HUMAIN's voice list, or its UUID |
| `model` | `str` | `"nebula"` | TTS model; `nebula` outputs 24 kHz mono |
| `api_key` | `str` | env `HUMAIN_API_KEY` | HUMAIN Voice API key |
| `api_url` | `str` | env `HUMAIN_API_URL` or `"https://api.voice.humain.com"` | API base URL |
| `tokenizer` | `tokenize.SentenceTokenizer` | `blingfire` | Splits streamed input into sentences; each sentence is one request |

Voices:

| Voice | Gender | Dialect |
|-------|--------|---------|
| `mul_nourah`, `mul_sara` | female | Saudi |
| `mul_abdullah`, `mul_abdulaziz` | male | Saudi |
| `mul_fadi` | male | Levantine |
| `mul_yassin` | male | Moroccan |
| `mul_youssef` | male | Egyptian |

All voices also speak English; Arabic script in the text selects Arabic. Change the voice during a session with `tts.update_options(voice="mul_sara")`.

## License

Apache 2.0
