# FonadaLabs plugin for LiveKit Agents

A LiveKit Agents plugin for the FonadaLabs voice AI APIs:

- **TTS** — Klone V2 Flash (streaming, default) and Klone V2 Pro
- **STT / ASR** — Kapture V3

## Features

- ⚡ **Low-latency streaming TTS**: Klone V2 Flash streams 24 kHz PCM over a pooled WebSocket — one connection is reused across agent replies
- 🎙️ **Voice Arena voices**: pick any voice by its `share_id`
- 🇮🇳 **Indian languages**: 12 TTS languages on Flash, 48 on Pro, 11 ASR languages on Kapture V3
- 🎛️ **Style control**: `instruct` (e.g. `"calm, friendly"`), `num_step`, `speed`
- 🔒 **Secure**: API key sent as a `Bearer` header

## Installation

```bash
pip install livekit-plugins-fonadalabs
```

## Pre-requisites

You'll need an API key from FonadaLabs (https://fonadalabs.ai). Set it as an environment variable: `FONADALABS_API_KEY`.

## Usage

```python
from livekit.agents import Agent
from livekit.plugins import fonadalabs, openai, silero

agent = Agent(
    instructions="You are a helpful voice assistant.",
    stt=fonadalabs.STT(model="v3", language="hi"),
    llm=openai.LLM(model="gpt-4o-mini"),
    tts=fonadalabs.TTS(
        model="klone_v2_flash",
        language="Hindi",
        share_id="UFNPsnEd",  # Voice Arena share_id
    ),
    vad=silero.VAD.load(),
)
```

## Text-to-Speech

| Model | Transport | Languages | Notes |
|---|---|---|---|
| `klone_v2_flash` (default) | WebSocket `wss://api.fonada.ai/v1/voice-clone/ws` | 12 | Streaming, lowest latency, ₹6 / 10K chars |
| `klone_v2_pro` | HTTPS `POST /v1/voice-clone/chunks` | 48 | Higher quality, no streaming, ₹60 / 10K chars |

### Parameters

| Argument | Description |
|---|---|
| `share_id` | **Required.** Voice Arena `share_id` of the voice (8 characters, e.g. `"UFNPsnEd"`). Voice names are not accepted |
| `model` | `"klone_v2_flash"` or `"klone_v2_pro"` |
| `language` | ISO code or display name (`"hi"` / `"Hindi"`) |
| `instruct` | Style hint, up to 500 characters |
| `num_step` | Diffusion steps, 8–64 (Flash only) |
| `speed` | 0.25–4.0 (FonadaLabs marks this as currently unreliable on Flash) |

Find a voice in Voice Arena and copy its `share_id`. An unknown `share_id` fails with `404 voice_not_found`.

Each agent reply is sent as **one** request (up to 5,000 characters; longer text is split at sentence boundaries). The server splits it by sentence and streams audio as it is generated. This keeps usage within the default rate limit of 10 requests per minute.

### Klone V2 Flash languages

| Code | Language | Code | Language |
|---|---|---|---|
| `bn` | Bengali | `mr` | Marathi |
| `en` | English | `or` | Odia |
| `gu` | Gujarati | `pa` | Punjabi |
| `hi` | Hindi | `ta` | Tamil |
| `kn` | Kannada | `te` | Telugu |
| `ml` | Malayalam | `ur` | Urdu |

### Klone V2 Pro languages

English (`en`), Chinese (`zh`), Spanish (`es`), French (`fr`), Standard Arabic (`arb`), Portuguese (`pt`), Russian (`ru`), Japanese (`ja`), Indonesian (`id`), German (`de`), Korean (`ko`), Turkish (`tr`), Vietnamese (`vi`), Italian (`it`), Thai (`th`), Polish (`pl`), Burmese (`my`), Persian (`fa`), Swahili (`sw`), Javanese (`jv`), Malay (`ms`), Filipino (`fil`), Dutch (`nl`), Ukrainian (`uk`), Romanian (`ro`), Hindi (`hi`), Bengali (`bn`), Tamil (`ta`), Telugu (`te`), Marathi (`mr`), Gujarati (`gu`), Kannada (`kn`), Malayalam (`ml`), Punjabi (`pa`), Urdu (`ur`), Assamese (`as`), Odia (`ory`), Maithili (`mai`), Bhojpuri (`bho`), Sanskrit (`sa`), Sindhi (`sd`), Kashmiri (`ks`), Dogri (`dgo`), Santali (`sat`), Manipuri (`mni`), Bodo (`brx`), Goan Konkani (`gom`), Konkani (`knn`).

## Speech-to-Text

| Model | Endpoint | Languages |
|---|---|---|
| `v3` — Kapture V3 (default) | `POST https://api.fonada.ai/v3/asr/transcribe` | 11 |

Recognition is VAD-buffered: LiveKit collects each utterance (using Silero VAD) and the plugin transcribes it with a single request.

### Kapture languages

| Code | Language | Code | Language |
|---|---|---|---|
| `en` | English | `mr` | Marathi |
| `bn` | Bengali | `or` | Odia |
| `gu` | Gujarati | `pa` | Punjabi |
| `hi` | Hindi | `ta` | Tamil |
| `kn` | Kannada | `te` | Telugu |
| `ml` | Malayalam | | |

## Documentation

- Klone V2 Flash: https://fonadalabs.ai/docs/text-to-speech/klone-v2-flash
- Klone V2 Pro: https://fonadalabs.ai/docs/text-to-speech/klone-v2-pro
- Kapture V3: https://fonadalabs.ai/docs/speech-to-text/kapture-v3
