# DeepL plugin for LiveKit Agents

Support for realtime speech-to-speech translation with the [DeepL Voice API](https://developers.deepl.com/docs/voice/overview).

The plugin is a simultaneous interpreter: it listens to one speaker and speaks their words in another language as they talk, transcribing both sides. It runs as a full-duplex model, so the translation plays over the speaker instead of waiting for them to finish.

## Installation

```bash
pip install livekit-plugins-deepl
```

## Pre-requisites

You need a DeepL API key with access to the Voice API. Set it in your environment:

```bash
DEEPL_API_KEY=<your_deepl_api_key>
```

`DEEPL_AUTH_KEY` is accepted as well. Keys ending in `:fx` use the DeepL API Free endpoint; all others use DeepL API Pro.

## Usage

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import deepl

session = AgentSession(
    llm=deepl.realtime.VoiceModel(
        source_language="en",
        target_language="de",
        voice="match",  # a voice close to the speaker's own
    ),
)
await session.start(agent=Agent(instructions=""), room=ctx.room)
```

`AgentSession` runs the model through `llm.DuplexRealtimeAdapter` for you. A runnable version is in [`examples/deepl_interpreter.py`](../../examples/other/translation/deepl_interpreter.py); try it with headphones, `python deepl_interpreter.py console`.

Each session translates one speaker into one language, which is the DeepL Voice limit for translated speech. To bridge a conversation, run one session per speaker and target language: [`examples/deepl_meeting.py`](../../examples/other/translation/deepl_meeting.py) interprets everyone in a room for everyone else.

### Options

- `target_language`: language to speak, as an IETF BCP 47 tag (`de`, `pt-BR`, ...).
- `source_language`: the speaker's language. When given, DeepL is held to it (`source_language_mode="fixed"`), which lowers latency; pass `source_language_mode="auto"` to use it as a hint only. Unset, DeepL detects the language.
- `voice`: `male`, `female`, or `match`. `match` falls back to a preset voice for languages whose speech DeepL does not synthesize itself.
- `formality`: `default`, `formal`/`more`, or `informal`/`less`.
- `glossary_ids`: DeepL glossaries to apply, highest priority first.
- `base_url`: the DeepL API to use, for example a staging endpoint; falls back to `DEEPL_BASE_URL`.

See [Supported Voice Languages](https://developers.deepl.com/docs/voice/supported-voice-languages) for language availability.

### Behavior

- The translation plays over the speaker and is never interrupted by them; each pause in the translated speech ends one agent message.
- The speaker's original words are recorded as user messages, the translation as assistant messages.
- The session configuration is fixed when it starts. Agent instructions and chat context are ignored, and tools raise an error.
- The interpreter only speaks translations, so `generate_reply()` (for example an `on_enter` greeting) fails with a `RealtimeError`.
- Dropped connections are resumed with DeepL's session tokens, keeping the translation context.
