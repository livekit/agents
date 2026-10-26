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

#### Voxtral 3

The Voxtral 3 model (`voxtral-mini-tts-3`) is the default. By default the text is synthesized over http, sentence
by sentence. The speech starts as soon as the first sentence of the LLM reply is complete.

```python
from livekit.plugins import mistralai

tts = mistralai.TTS(model="voxtral-mini-tts-3", voice="en_paul_neutral")

# Voxtral 3 adds multilingual voices (`voice_<n>`, see `TTSVoxtral3Voices`) and still accepts the older ones.
# The older models only accept the older voices, anything else raises a ValueError.
tts = mistralai.TTS(model="voxtral-mini-tts-3", voice="voice_447")  # Margaret, en-US

# How expressive the speech is, between 1 and 5 (Voxtral 3 only)
tts = mistralai.TTS(model="voxtral-mini-tts-3", expressiveness=3)

# Pronunciation hints, as IPA phones per word (Voxtral 3 only, http only)
tts = mistralai.TTS(phoneme_hints={"tomato": ["t", "ə", "m", "ɑ", "t", "oʊ"]})
```

##### Realtime websocket

With `use_realtime=True` the speech is streamed over the realtime websocket api while the LLM generates the text,
using a pool of connections reused between the requests. In our benchmarks it shortens the total time of long
replies, but the speech starts later than with http when the text comes token by token from an LLM, and
`phoneme_hints` isn't supported.

```python
tts = mistralai.TTS(use_realtime=True)
```

The streamed speech is always PCM, `response_format` only applies to the http mode. The realtime engine can't
handle a single request of more than about 3,500 characters, so a longer reply is automatically split in several
requests.

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
