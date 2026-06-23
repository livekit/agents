# rumik-ai plugin for LiveKit Agents

Support for low-latency text-to-speech with rumik-ai Silk models.

The plugin streams raw PCM audio from rumik-ai's one-shot WebSocket sessions and exposes
the provider through the standard LiveKit Agents TTS interface.

## Installation

```bash
pip install livekit-plugins-rumik-ai
```

## Pre-requisites

You'll need a rumik-ai API key. It can be set as an environment variable:

```bash
export RUMIK_API_KEY=...
```

## Usage

Use Muga for low-latency Hinglish speech. In the normal agent path, have the LLM
include the global tone tag at the start of every reply:

```python
from livekit.plugins import rumik_ai

tts = rumik_ai.TTS(model="muga")
```

Valid Muga input must then look like:

```text
[happy] Arre yaar tu aa gaya, kab se wait kar rahi thi main!
[sad] <sigh> Theek hai, samajh sakti hoon.
```

Use Mulberry for description or preset speaker-based synthesis:

```python
from livekit.plugins import rumik_ai

tts = rumik_ai.TTS(
    model="mulberry",
    description="warm, upbeat narrator",
    speaker="speaker_2",
    f0_up_key=0,
)
```

Mulberry speaks pure English and Hindi; write Hindi in Devanagari (keeping English words in
Latin script) rather than the Romanized Hinglish that Muga uses. It does not use Muga tone
tags or Muga event tags in `text`. Voice style is controlled by `description`, and preset
voices are selected with `speaker`. These (plus `f0_up_key` and the sampling params) are
sent per request, so the voice can be changed between turns with `update_options()` without
reconnecting.
By default Mulberry streams sentence-by-sentence for lower latency; pass
`full_response_aggregation=True` to synthesize the whole reply in one request for more
consistent prosody across sentences.

With an `AgentSession`:

```python
from livekit.agents import AgentSession, inference
from livekit.plugins import rumik_ai

session = AgentSession(
    stt=inference.STT(model="deepgram/nova-3", language="multi"),
    llm=inference.LLM(model="openai/gpt-4.1-mini"),
    tts=rumik_ai.TTS(model="muga"),
)
```

## Muga prompting

Muga requires one global tone for every utterance. The recommended voice-agent flow is
to instruct the LLM to choose and prepend the tone tag itself, then pass the tagged text
directly to rumik-ai TTS.

For direct TTS smoke tests, you can pass `tone="happy"` or another supported tone as a
fallback. In that mode, the plugin prefixes untagged input with the configured tone and
requires any existing tag to match that configured tone.

Supported tones:

- `happy`
- `excited`
- `sad`
- `angry`
- `neutral`
- `whisper`

Supported inline events:

- `<laugh>`
- `<chuckle>`
- `<sigh>`

The plugin validates tone and event combinations before sending text to Rumik AI so invalid
prompts fail early.

Common examples:

```python
rumik_ai.TTS(model="muga")  # LLM or caller must include [tone]
rumik_ai.TTS(model="muga", tone="happy")  # fallback for untagged direct TTS text
```

Valid Muga input:

```text
[happy] Arre yaar tu aa gaya, kab se wait kar rahi thi main!
[excited] <laugh> Bhai jeet gaye! Mujhe abhi tak vishwas nahi ho raha!
```

With `tone="happy"` configured, untagged direct input like:

```text
Arre yaar tu aa gaya, kab se wait kar rahi thi main!
```

is sent as `[happy] Arre yaar tu aa gaya, kab se wait kar rahi thi main!`.

## Mulberry controls

Mulberry supports natural-language voice descriptions and optional preset speakers:

```python
tts = rumik_ai.TTS(
    model="mulberry",
    description="calm female narrator",
    speaker="speaker_1",
    f0_up_key=0,
)
```

Supported preset speakers are `speaker_1`, `speaker_2`, `speaker_3`, and `speaker_4`.
`f0_up_key` accepts values from `-12` to `12`.

Mulberry speaks pure English and Hindi. Write English in the Latin alphabet and Hindi in
Devanagari (keep any English words inside a Hindi sentence in Latin script). The
Romanized/Latin "Hinglish" that Muga expects is **not** how you prompt Mulberry — give it
Devanagari for Hindi rather than transliteration.

Valid Mulberry input:

```text
Hey, I'm Mira. How was your day today?
आज का din कैसा रहा? कुछ heavy लग रहा है क्या?
```

Invalid Mulberry input (Muga tone and event tags are not supported):

```text
[happy] Arre yaar tu aa gaya.
Arre <laugh> yaar.
```

## Streaming behavior

The plugin keeps one reusable rumik-ai WebSocket session (pooled across turns) and streams
the returned 24 kHz mono PCM frames back to LiveKit. The unit of synthesis is model-aware:
**muga** buffers the whole assistant reply into one request so its leading `[tone]` tag
conditions the entire utterance, while **mulberry** streams sentence-by-sentence for lower
time-to-first-word. Override per call with `full_response_aggregation`.

On a barge-in (the user interrupts), the plugin sends an explicit cancel to rumik-ai so the
in-flight generation stops immediately, and keeps the pooled WebSocket warm so the next
utterance avoids a reconnect.
