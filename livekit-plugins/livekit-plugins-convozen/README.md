# LiveKit Plugins ConvoZen

Agent Framework plugin for [ConvoZen](https://convozen.ai/) voice models:
**Akshara** for speech-to-text and **Ragini** for text-to-speech.

Both cover nine Indian languages — Bengali, English, Gujarati, Hindi, Kannada,
Malayalam, Marathi, Tamil, Telugu — including code-mixed speech.

ConvoZen's research write-ups describe the models themselves:
[Akshara](https://convozen.ai/research/article/akshara) for recognition and
[Ragini](https://convozen.ai/research/article/ragini) for synthesis.

## Installation

```bash
pip install livekit-plugins-convozen
```

## Pre-requisites

You'll need an API key from [ConvoZen](https://app.convozen.ai/developers/models). It can be set as an environment variable:
`CONVOZEN_API_KEY`

To reach a self-hosted deployment, pass `base_url=` or set `CONVOZEN_BASE_URL` —
the same variable the ConvoZen SDK reads.

## Usage

Akshara and Ragini are both batch models — a whole utterance in, a whole
transcript out; a whole sentence in, whole audio out. Neither streams
incrementally, so **a VAD is required on the session**. LiveKit then wraps the STT
in a `StreamAdapter` (the VAD segments speech into utterances) and the TTS in a
`StreamAdapter` (sentences are synthesized as the LLM produces them). Without a
VAD, `AgentSession` raises at startup.

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import convozen, openai, silero

session = AgentSession(
    stt=convozen.STT(language="hi"),
    llm=openai.LLM(model="gpt-5.6-luna"),
    tts=convozen.TTS(voice="roohi", language="hi"),
    vad=silero.VAD.load(),  # required — see above
)

await session.start(agent=Agent(instructions="You are a helpful assistant."), room=ctx.room)
```

### Speech-to-text

```python
convozen.STT(
    language="hi",  # reported on every transcript; also the default lang hint
    model="akshara-pro",  # or the base "akshara"
    lang_tags=["hi", "en"],  # explicit hints for code-mixed speech
    keywords=["ConvoZen", "Akshara"],  # vocabulary boosting
    word_timestamps=True,  # per-word timings on SpeechData.words
)
```

Recognition is a single HTTP round-trip per utterance, so it resolves after the
speaker stops rather than during. Use `keywords` for names and jargon the model
would otherwise render phonetically — it helps but is not a guarantee, so check
the terms that matter to you.

Akshara returns a `score` alongside the transcript. It is an unbounded
log-probability (closer to zero being better), not a `[0, 1]` confidence, so it is
exposed as `SpeechData.metadata["score"]` and `SpeechData.confidence` is left unset.

Keyterms configured on the `AgentSession` are forwarded as `keywords`, merged with
any passed to the constructor.

`word_timestamps` returns one entry per word. Timings are measured from the start
of each recognized utterance, not from the start of the session, so the STT does not
declare `aligned_transcript`. The start times track the audio, but the spans are
narrower than the spoken words, so treat `start_time` as meaningful and the width as
not.

### Text-to-speech

```python
convozen.TTS(
    voice="roohi",  # any voice id the account has access to
    model="ragini-v1",  # or the lighter "ragini-lite"
    language="hi",
    speed=1.0,
    # sample_rate defaults to the model's native rate:
    # 24000 for ragini-v1, 22050 for ragini-lite
)
```

Voices: `roohi`, `amaya`, `kiyansh`, `neeraj`, `manya`, `nidhi`, `ira`, `trisha`,
`charvi`. Any voice id the account has access to is accepted, so voices added
server-side work without a plugin upgrade.

Audio is streamed back as it is generated, which is what keeps time-to-first-audio
down. Pass `stream_response=False` to receive one complete WAV instead.

Sample rates of 22050 and 24000 Hz are supported; the default is
the model's native rate.
