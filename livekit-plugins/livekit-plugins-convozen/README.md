# LiveKit Plugins ConvoZen

Agent Framework plugin for [ConvoZen](https://convozen.ai/) voice models:
**Akshara** for speech-to-text and **Ragini** for text-to-speech.

Akshara covers nine Indian languages — Bengali, English, Gujarati, Hindi, Kannada,
Malayalam, Marathi, Tamil, Telugu — including code-mixed speech. Ragini 2.0 covers
English, Hindi, Kannada, Tamil and Telugu.

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
    tts=convozen.TTS(voice="sherin", language="hi"),
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
    voice="sherin",  # defaults to the language group's default voice
    model="ragini-v2",
    language="hi",  # en, hi, kn, ta or te
    # sample_rate defaults to 48000, the rate ragini-v2 returns
)
```

Ragini 2.0 runs a separate model for each language group, and each has its own
voices:

| Languages | Voices | Default |
|-----------|--------|---------|
| `en`, `hi` | `shrajay`, `siddharth`, `parmi`, `sherin`, `bhavya`, `shreya` | `shrajay` |
| `kn`, `ta`, `te` | `amaya`, `parmi`, `roohi`, `bhavya`/`bhagya`, `shrajay`, `siddharth` | `amaya` |

Pick a voice from the group that matches `language`. The server does not reject
a voice from the other group, or a misspelled one — it returns an arbitrary voice
instead. If you don't pass `voice`, the group's default is used, and it follows
`language` when that changes through `update_options`. Any voice id is accepted,
so voices added server-side work without a plugin upgrade.

`speed` is accepted but currently ignored by `ragini-v2`, which always speaks at
the natural pace.

Audio is streamed back as it is generated, which is what keeps time-to-first-audio
down. Pass `stream_response=False` to receive one complete WAV instead.

`ragini-v2` produces 48 kHz audio, and the plugin always requests that rate.
Passing a different `sample_rate` resamples it on the client.
