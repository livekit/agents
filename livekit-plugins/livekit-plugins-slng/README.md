# SLNG plugin for LiveKit Agents

Support for [SLNG](https://slng.ai/)'s voice AI gateway in LiveKit Agents, providing access to multiple STT and TTS providers through a unified API.

See [https://docs.slng.ai/](https://docs.slng.ai/) for more information.

## Installation

```bash
pip install livekit-plugins-slng
```

## Pre-requisites

You'll need an API key from SLNG and the host of the SLNG region you connect to, such as `us-east.api.slng.ai`. Set them as the environment variables `SLNG_API_KEY` and `SLNG_BASE_URL`, or pass them as `api_key` and `slng_base_url`. There is no default region.

## Usage

Pass an SLNG model identifier; the plugin connects through SLNG's Unmute Bridge and builds the endpoint itself.

```python
from livekit.plugins import slng

stt = slng.STT(
    model="deepgram/nova:3",
    language="en",
    slng_base_url="us-east.api.slng.ai",  # your region's host, or set SLNG_BASE_URL
)

tts = slng.TTS(
    model="deepgram/aura:2",
    voice="aura-2-thalia-en",  # provider voice ID, required
    slng_base_url="us-east.api.slng.ai",
    # language="en",           # optional; omit to use the model's catalog default
)
```

Additional keyword arguments are forwarded to the gateway and applied according to the selected model's contract. Failover across multiple models or endpoints is available via `connections=[...]`; see [docs.slng.ai](https://docs.slng.ai/) for details.

## TTS init fields

The plugin sends only the settings you set. `encoding` (always `linear16`) and `sample_rate` are always sent; `encoding` cannot be overridden, because the plugin decodes the audio itself. `language` and `speed` are sent only when you pass them, so the model's catalog defaults apply otherwise. Any other keyword argument is forwarded verbatim in the init `config`.

## TTS text chunking

`text_chunking` controls how LLM text is cut into frames for the gateway:

- `"sentence"`: one frame per sentence.
- `"phrase"`: words re-batched at `. ! ? , ; :` or every `phrase_max_chars` (60).
- `"word"`: one frame per word.

The default, `"auto"`, is `"sentence"`, or `"phrase"` when `word_tokenizer` is a `WordTokenizer`.

Sentence mode needs no setup and no language setting. The default `slng.SentenceTokenizer` ends a sentence at any script's terminator (`. ! ?`, the danda, the ideographic full stop, and the rest of Unicode's `Sentence_Terminal` set). Any piece longer than 200 characters is cut at a space, which is what makes a script with no terminator stream at all. Pass `word_tokenizer=slng.SentenceTokenizer(max_chars=...)` to change that length; an overriding tokenizer must be a `SentenceTokenizer` in this mode.

The plugin sends the opening of a long first sentence as soon as it exists, so a model that can start on part of a sentence begins speaking sooner, and every other model hears the sentence as before. This needs no configuration, and it happens only where the gateway supports it. Text in a script with no sentence terminator stays in one frame until it reaches `max_chars`, so first audio waits for the whole reply unless you lower it.

## TTS connections

The plugin holds one WebSocket per call. It sends `init` once, then one `text` frame per sentence followed by a `flush` that ends the reply, and keeps the socket open for the next reply, reconnecting if the gateway closes it. With `connections=[...]`, only the model in use holds a connection.

`warm_standby_enabled` is on by default: `prewarm()` opens the connection before the first reply, and the plugin reopens it in the background if the gateway closes it. That connection counts as one concurrent session on your key for the whole call, including silences. After five idle minutes the plugin closes it, and the next reply opens a new one; a connection open for 20 minutes is replaced between replies. A TTS that an agent handoff replaces closes its connection 10 seconds after its session stops using it. Set `warm_standby_enabled=False` to open a connection for each reply and close it afterwards: nothing is held between replies, and every reply pays a connect. A reply that starts while the previous one is still being cancelled, and `synthesize()`, each use their own short-lived socket.

## End of turn finalization

For the lowest STT turn latency, let the plugin know when the user stops speaking. The plugin then sends a finalize signal so the provider returns the final transcript immediately instead of waiting for its own endpointing:

```python
session = AgentSession(stt=stt, ...)
stt.attach_to_session(session)
```

Or wire it manually:

```python
@session.on("user_state_changed")
def _on_user_state_changed(ev):
    stt.notify_user_state(ev.new_state)
```

Without this hook the plugin still works, but end of turn detection relies entirely on the provider's endpointing, which typically adds a few hundred milliseconds per turn.

## Region override

The plugin supports gateway region routing via the `region_override` option on both `STT` and `TTS`.
This maps directly to the gateway's `X-Region-Override` header.
See the available regions at [docs.slng.ai/region-override](https://docs.slng.ai/region-override).

You can pass either a single region:

```python
stt = slng.STT(
    api_key="your-slng-api-key",
    model="deepgram/nova:3",
    region_override="eu-west-1",
)
```

Or multiple preferred regions in priority order:

```python
tts = slng.TTS(
    api_key="your-slng-api-key",
    model="deepgram/aura:2",
    voice="aura-2-thalia-en",
    region_override=["eu-west-1", "us-east-1"],
)
```

To constrain routing to a broad geographic zone instead of a specific region,
use `world_part_override` (for example `"eu"`), which maps to the gateway's
`X-World-Part-Override` header. `region_override` takes precedence when both
are set.

## Migrating from 1.x

Version 2.0 is a breaking change:

- All traffic goes through the Unmute Bridge. `model_endpoint` and `model_endpoints` were removed; pass a model identifier or `connections=[...]` instead.
- STT no longer defaults to `model="deepgram/nova:3"`; pass a model (or `connections`) explicitly.
- TTS `voice` is required and passed verbatim as the provider's voice identifier.
- Language codes are no longer normalized client-side; send the value the model expects (for example BCP-47 `hi-IN` for Sarvam, not `hi`).
- STT `recognize()` (HTTP batch) is no longer supported; use `stream()`. Only `pcm_s16le` input audio is supported.
- `api_token` still works on STT but is deprecated; use `api_key`.
- TTS no longer sends `language="en"` when `language` is omitted; the model's catalog default applies instead.
- TTS `text_chunking` defaults to `"sentence"`: one frame per sentence, rather than clause-sized frames. A `word_tokenizer` that is a `WordTokenizer` keeps the clause-sized frames.
- TTS `warm_standby_enabled` defaults to True, so the connection is open from session start and counts as one concurrent session for the whole call. Set it to False for a connection per reply, as before.
- `slng_base_url` has no default on STT or TTS: pass your region's host, for example `us-east.api.slng.ai`, or set `SLNG_BASE_URL`.
