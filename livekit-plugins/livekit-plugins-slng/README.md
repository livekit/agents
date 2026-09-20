# SLNG plugin for LiveKit Agents

Support for [SLNG](https://slng.ai/)'s voice AI gateway in LiveKit Agents, providing access to multiple STT and TTS providers through a unified API.

See [https://docs.slng.ai/](https://docs.slng.ai/) for more information.

## Installation

```bash
pip install livekit-plugins-slng
```

## Pre-requisites

You'll need an API key from SLNG. It can be set as an environment variable: `SLNG_API_KEY`

## Usage

Pass an SLNG model identifier; the plugin connects through SLNG's Unmute Bridge and builds the endpoint itself.

```python
from livekit.plugins import slng

stt = slng.STT(
    model="deepgram/nova:3",
    language="en",
)

tts = slng.TTS(
    model="deepgram/aura:2",
    voice="aura-2-thalia-en",  # provider voice ID, required
    # language="en",           # optional; omit to use the model's catalog default
)
```

Additional keyword arguments are forwarded to the gateway and applied according to the selected model's contract. Failover across multiple models or endpoints is available via `connections=[...]`; see [docs.slng.ai](https://docs.slng.ai/) for details.

## TTS init fields

The plugin sends only the settings you set. `encoding` (always `linear16`) and `sample_rate` are always sent, because the plugin needs them to decode the audio it receives. `language` and `speed` are sent only when you pass them, so the selected model's catalog defaults apply otherwise. Any other keyword argument is forwarded verbatim in the init `config`.

## TTS text chunking

`text_chunking` controls how LLM text is cut into frames for the gateway:

- `"sentence"` (the default, and what `"auto"` resolves to): one frame per complete sentence, using `tokenize.blingfire.SentenceTokenizer`. Pass your own `word_tokenizer` to change the tokenizer.
- `"phrase"`: words re-batched at `. ! ? , ; :` or every `phrase_max_chars` (60). This was the behaviour before sentence mode existed.
- `"word"`: one frame per word.

Because the plugin sends complete sentences, a provider may run in per-frame mode (`segment="immediate"` on Rime, `auto_mode=True` on ElevenLabs) or in its own buffering mode, and both sound the same. That setting no longer affects audio quality, only how the provider paces its work.

First audio arrives once the first sentence is complete, on every provider. Keep the opening sentence of each reply short ("Got it." then the rest) so the first frame leaves early. This is standard voice-agent practice, and it is what keeps latency flat with sentence-sized frames.

## TTS connections

The plugin holds one WebSocket per call. It sends `init` once, then one `text` frame per sentence with `flush: true` on the reply's last frame, waits for `audio_end`, and keeps the socket open for the next reply. If the gateway closes the socket after a reply, the plugin reconnects and replays that reply, so a gateway that ends the session after every reply still works.

With `connections=[...]` only the model currently in use holds a connection. Switching to a fallback closes the previous model's socket, so a failover chain does not hold one socket per model.

That connection counts as one concurrent session on your key for the whole call, including silences. Two short-lived exceptions: a reply that starts while the previous one is still being cancelled opens its own socket for that reply, and `synthesize()` (non-streaming) always uses a dedicated one.

Use the regional `<region>.api.slng.ai` base URLs. They expect the `flush` flag on the final text frame, which is the form the bridge contract defines. The older `api.slng.ai` host honours that flag for some providers only. On that host the symptom is a reply that plays to the end and then hangs until the connection timeout; in the segment log `first_audio_ms` is set and `audio_end_ms` is null. Move to a regional base URL, which needs a new API key.

`warm_standby_enabled` is on by default and means "connect at session start". `prewarm()` opens the connection before the first reply, and the plugin reopens it in the background when the gateway closes a socket that has carried text. A socket the gateway closes before any reply used it is retried at most every two seconds, so a gateway that accepts and immediately closes cannot become a connect storm. Set `warm_standby_enabled=False` to connect on the first reply instead.

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
