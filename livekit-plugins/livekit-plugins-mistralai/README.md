# Mistral AI Plugin for LiveKit Agents

Support for Mistral AI STT, TTS, and LLM services.

## Installation

```bash
pip install livekit-plugins-mistralai
```

For streaming STT (Voxtral Realtime), also install `silero` plugin.

```bash
pip install livekit-plugins-silero
```

## Pre-requisites

You'll need an API key from Mistral AI. It can be set as an environment variable:

```bash
export MISTRAL_API_KEY=your_api_key_here
```

## Usage

### Speech-to-Text (STT)

#### Offline transcription

```python
from livekit.plugins import mistralai

stt = mistralai.STT()

# With context biasing
stt = mistralai.STT(
    model="voxtral-mini-latest",
    context_bias=["LiveKit", "Voxtral", "Mistral"]
)
```

#### Realtime streaming transcription

Voxtral Realtime streams interim transcripts over a WebSocket connection. Since this
model has no server-side endpointing, the plugin runs an internal Silero VAD to detect
when the user stops speaking and flush the audio — producing final transcripts and
driving the end-of-turn pipeline.

```python
from livekit.plugins import mistralai
from livekit.plugins.silero import VAD

# Using Silero VAD with default settings (550ms silence threshold)
stt = mistralai.STT(model="voxtral-mini-transcribe-realtime-2602")

# Using custom VAD settings (e.g. shorter silence threshold for faster responses)
stt = mistralai.STT(
    model="voxtral-mini-transcribe-realtime-2602",
    vad=VAD.load(min_silence_duration=0.3),
)
```

#### Self-hosted Voxtral Realtime (vLLM)

`SelfHostedSTT` streams to Voxtral Realtime on your own vLLM server, over vLLM's
`/v1/realtime` WebSocket. No Mistral API key is needed.

```bash
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602
```

```python
from livekit.agents import AgentSession
from livekit.plugins import mistralai, silero

session = AgentSession(
    stt=mistralai.SelfHostedSTT(base_url="http://gpu-box:8000"),
    vad=silero.VAD.load(),
    # llm=..., tts=...
)
```

Each user turn is one transcription run. The session's VAD or turn detector ends the turn,
which flushes the STT and produces the final transcript, so give the session a `vad` (or
pass one to `SelfHostedSTT(vad=...)` when it has none). Pass `api_key` if the server was
started with `--api-key`.

A run holds one token of context per 80 ms of audio, silence between turns included. Set
`max_model_len` to the server's `--max-model-len` (default 2048, about 164 s): the STT ends a
long run at a pause before it fills, and starts the next one without dropping audio.

### Text-to-Speech (TTS)

```python
from livekit.plugins import mistralai

# Using a built-in voice
tts = mistralai.TTS(voice="en_paul_neutral")

# Using zero-shot voice cloning
import base64
ref_audio_b64 = base64.b64encode(open("sample.mp3", "rb").read()).decode()
tts = mistralai.TTS(ref_audio=ref_audio_b64)
```

### LLM

```python
from livekit.plugins import mistralai

llm = mistralai.LLM()

# With all available options
llm = mistralai.LLM(
    model="mistral-large-latest",
    temperature=0.7,
    top_p=0.9,
    max_completion_tokens=150,
    presence_penalty=0.1,
    frequency_penalty=0.1,
    random_seed=42,
    tool_choice="auto",
)

# With provider tools
agent = Agent(
    llm=llm,
    tools=[
        mistralai.tools.WebSearch(),
        mistralai.tools.CodeInterpreter(),
        mistralai.tools.DocumentLibrary(library_ids=["<your-library-id>"]),
        mistralai.tools.Connector(connector_id="<your_connector_id>")
    ]
)
```
