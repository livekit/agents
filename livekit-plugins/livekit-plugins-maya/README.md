# Maya Research voice models for LiveKit Agents

A native LiveKit TTS plugin for the [Maya Research API](https://www.mayaresearch.ai/llm.txt).
The provider and package names are model-independent.

The current documented model, checked on 8 September 2026, is **Maya Calyx**.
The default voice is **Aarav**. The public API documents Hindi, Telugu, Indian
English, Tamil, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia and Punjabi.
Model and voice strings are passed to the service, so a future model does not
need a renamed integration. Check model/voice compatibility in the current API
reference before changing them.

## Install this contribution

This contribution is not yet a published upstream package. From a checkout of
this branch, install the plugin into your agent environment:

```sh
uv pip install --no-sources ./livekit-plugins/livekit-plugins-maya
```

This resolves released LiveKit Agents rather than the repository's development
workspace. The [Maya Research Cookbook](https://github.com/MayaResearch/maya-cookbook)
provides a tested immutable pin, API reference, TTS quickstarts, and complete
LiveKit, Pipecat and from-scratch agent examples.

## Configure

Set `MAYA_API_KEY` in your server environment. Do not put it in browser code,
URLs, source control, logs or coding-agent prompts.

Custom `base_url` / `MAYA_BASE_URL` values must use HTTPS or WSS. Plain HTTP/WS,
including loopback URLs, is rejected before any key or text is transmitted.

```python
from livekit.agents import AgentSession
from livekit.plugins import maya

session = AgentSession(
    # Supply your application's STT, LLM and turn-handling configuration.
    tts=maya.TTS(model="Maya Calyx", voice="Aarav", language="hi"),
)
```

Omit `language` (or use `None`) for mixed-language input. To restore that mode
after choosing a language, call `update_options(language=None)` on your TTS
instance. Omitting the argument in `update_options` leaves the setting unchanged;
an active turn retains its existing settings. A Maya key provides TTS only, not
speech recognition, an LLM, or LiveKit room credentials.

## Streaming contract

- One persistent WebSocket can serve multiple completed turns.
- Each turn uses a new context ID. Sentences share that ID and one final closer.
- Follow current LiveKit semantics: create a new `stream()` per segment. Push
  incremental text, then call `end_input()`. Do not push new text after `flush()`.
- No text is sent before validated startup metadata. Output is 24 kHz mono
  signed 16-bit little-endian PCM; other formats fail instead of sounding wrong.
- Base64 is decoded strictly. Split sample bytes and the final partial frame
  are preserved. Late frames from another context are ignored.
- Slow first text does not consume the server response timeout. After audio
  progresses, a pause awaiting more LLM text does not abort the open turn.
  New text re-arms the progress timeout; the final closer starts a bounded
  audio/end wait. Text sends themselves are bounded too. Empty turns generate
  neither speech nor a nonexistent turn-closer.
- Cancellation stops the turn and discards its connection; the next turn
  cannot inherit abandoned audio. LiveKit handles clearing local playout.
- Updating options selects a correctly configured connection for the next turn,
  without closing a currently active turn.
- Closing the provider prevents new acquisitions and retires handshakes that
  finish during shutdown, including directly constructed public streams.
- Errors after audio receipt are not automatically retried, avoiding repeated
  speech. Authentication and malformed protocol errors are not retried either.

The default LiveKit BlingFire tokenizer primarily splits western punctuation.
Pass an appropriate `tokenizer=` when early danda-delimited sentence emission
is required. This plugin does not rewrite, normalize, or translate input text.

The v2 protocol has no per-sentence completion acknowledgements. Once audio has
arrived while text input remains open, an idle provider cannot be distinguished
from one waiting for the LLM. The progress timeout therefore resumes on new text
or `end_input()`. Applications should also bound the LLM/overall turn and always
end or cancel an abandoned input stream. No timeout policy can prove that every
word was spoken; that needs end-to-end transcription or listening.

## Development and tests

From the LiveKit repository root:

```sh
uv sync --package livekit-plugins-maya --group dev --no-group typing --python 3.12 --locked
uv run --no-sync pytest tests/test_maya_tts.py --unit -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy --follow-imports=silent livekit-plugins/livekit-plugins-maya/livekit/plugins/maya
```

The plugin tests use an in-memory protocol fixture, with no network or credentials.
Live API tests require explicit authorization and synthetic inputs. Successful
audio receipt is not a human listening score or a physical microphone/room test.
