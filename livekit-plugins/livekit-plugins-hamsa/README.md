# LiveKit Plugins Hamsa

**Hamsa** text-to-speech for [LiveKit Agents](https://docs.livekit.io/agents/).

[Hamsa](https://tryhamsa.com) builds Arabic speech models for the way Arabic is spoken: Egyptian, Levantine (Lebanese, Syrian, Jordanian, Palestinian), Gulf (Saudi, Emirati, Kuwaiti, Qatari, Bahraini, Omani), Iraqi and Modern Standard Arabic, plus English. See the [API docs](https://docs.tryhamsa.com).

LiveKit docs: [text-to-speech](https://docs.livekit.io/agents/models/tts/hamsa/) (live with the next release).

## Installation

```bash
pip install livekit-plugins-hamsa
```

Set `HAMSA_API_KEY`, or pass `api_key=` to `TTS`. Create a key in the [Hamsa dashboard](https://dashboard.tryhamsa.com).

## Usage

### In an agent

```python
from livekit.agents import AgentSession
from livekit.plugins import hamsa

session = AgentSession(
    stt=...,
    llm=...,
    tts=hamsa.TTS(speaker="Mariam", dialect="egy"),
    vad=...,
)
```

### Text-to-speech

`TTS` talks to `WS /v1/realtime/ws`. Each request carries one complete text and the audio streams back as raw 16-bit PCM, so `stream()` splits the agent's text into sentences and sends one request per sentence on a pooled connection; `synthesize()` sends the whole text in one request. `prewarm()` opens a connection ahead of the first request. Hamsa accepts up to 2,000 characters per request; longer text is split at word boundaries into several requests on the same connection.

```python
tts = hamsa.TTS(speaker="Carla", dialect="leb")
async with tts:
    stream = tts.stream()
    stream.push_text("أهلا وسهلا فيك. كيف فيني ساعدك اليوم؟")
    stream.end_input()
    async for audio in stream:
        print(f"Got audio: {audio.frame.duration:.2f}s")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | env `HAMSA_API_KEY` | Hamsa API key |
| `speaker` | `str` | `"Mariam"` | Pre-built voice name, or the UUID of a cloned voice. Use a speaker recorded in the chosen `dialect` |
| `dialect` | `str` | `"egy"` | `egy`, `leb`, `syr`, `jor`, `pls`, `irq`, `ksa`, `uae`, `kuw`, `qat`, `bah`, `oma`, `ar-sa` (Gulf), `msa` or `en` |
| `sample_rate` | `int` | `16000` | `16000` or `8000` (telephony) |
| `expressiveness` | `float \| None` | Hamsa's default (`1`) | `0` (flat) to `2` (very expressive) |
| `tokenizer` | `tokenize.SentenceTokenizer` | `blingfire` | Splits streamed input into sentences; each sentence is one request |
| `base_url` | `str` | `"wss://api.tryhamsa.com"` | API base URL |
| `http_session` | `aiohttp.ClientSession` | job HTTP context | Shared session for the WebSocket |

Example voices per dialect (the full list is in the Hamsa dashboard):

| Dialect | Code | Voices |
|---------|------|--------|
| Egyptian | `egy` | Mariam, Samir |
| Lebanese | `leb` | Carla, Majd |
| Syrian | `syr` | Dalal, Mais |
| Jordanian | `jor` | Lana, Jasem |
| Palestinian | `pls` | Layan, Amjad |
| Saudi | `ksa` | Hiba, Fahd |
| Emirati | `uae` | Salma, Dima |
| Modern Standard Arabic | `msa` | Salem, Tamim |

Change the voice during a session with `tts.update_options(speaker="Dalal", dialect="syr")`.

## License

Apache 2.0
