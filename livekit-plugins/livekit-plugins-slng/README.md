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

## Init fields

On TTS, the plugin sends only the settings you set. `model` and `voice` are always sent, the model from the connection and the voice from `voice` (or the connection's own `voice`). `encoding` (always `linear16`) and `sample_rate` are always sent too; `encoding` cannot be changed, because the plugin decodes the audio itself. `language` and `speed` are sent only when you pass them, so the model's catalog defaults apply otherwise. Any other keyword argument is forwarded verbatim in the init `config`.

A connection can carry its own init message, as in `TTSConnectionConfig(init=...)`. On TTS, every field in it is kept, but the plugin's settings above win wherever both set one, so `update_options` still applies. On STT, where every setting has a default, a connection's init is sent as written, except for the options you later change with `update_options`, which are set in its `config` and in any top-level copy of the same field.

## TTS text chunking

`text_chunking` controls how LLM text is cut into frames for the gateway:

- `"sentence"`: one frame per sentence.
- `"phrase"`: words re-batched at `. ! ? , ; :` or every `phrase_max_chars` (60).
- `"word"`: one frame per word.

The default, `"auto"`, is `"sentence"`, or `"phrase"` when `word_tokenizer` is a `WordTokenizer`.

Sentence mode needs no setup and no language setting. The default `slng.SentenceTokenizer` ends a sentence at any script's terminator (`. ! ?`, the danda, the ideographic full stop, and the rest of Unicode's `Sentence_Terminal` set). A line break also ends a piece, so a heading or a list item leaves as its own frame, and a list number stays with its line. A line starting with a lowercase letter, as in hard-wrapped text, continues the sentence above it, unless the line above ends with a colon or the new line starts with a list letter such as "a)". Greek questions end at their `;`, Chinese or Japanese written with ASCII stops splits at them, and Indonesian honorifics such as "Bpk." and "Kec." stay with the name after them. Any piece longer than 200 characters is cut at a space, which is what makes a script with no terminator stream at all. Pass `word_tokenizer=slng.SentenceTokenizer(max_chars=...)` to change that length. In this mode an overriding tokenizer must be a livekit `SentenceTokenizer`, and only `slng.SentenceTokenizer` releases openings early. Text with no letter at the end of a reply, such as a number on its own last line, joins the sentence before it, since some models refuse a frame with no letter; only a run of it longer than that length is sent on its own.

Where the gateway supports it, the plugin also sends the opening of a sentence at the start of a reply as soon as it reaches a clause break, so a model that can start on part of a sentence begins speaking sooner, and every other model hears the sentence as before. After a short first sentence such as "Sure.", the second may open early too. This needs no configuration. An opening is at least 25 characters, which keeps a short reply such as "Hi there, how can I help?" in one frame, and so cacheable. For a model that renders every frame as its own utterance, such as Rime Coda with `segment="immediate"`, pass `word_tokenizer=slng.SentenceTokenizer(partial_head_min_chars=8)` so that speech starts at the first comma, as it did with earlier releases; an opening that short is followed by one more at the next clause break. Text in a script with no sentence terminator stays in one frame until it reaches `max_chars`, apart from such an opening, so first audio can wait for most of the reply unless you lower it.

## TTS connections

The plugin holds one WebSocket per call. It sends `init` once, then one `text` frame per sentence followed by a `flush` that ends the reply, and keeps the socket open for the next reply, reconnecting if the gateway closes it. With `connections=[...]`, only the model in use holds a connection.

`warm_standby_enabled` is on by default: `prewarm()` opens the connection before the first reply, and the plugin reopens it in the background if the gateway closes it or a reply fails on it. That connection counts as one concurrent session on your key for the whole call, including silences. After five idle minutes the plugin closes it, and the next reply opens a new one; a connection open for 20 minutes is replaced between replies. A TTS that an agent handoff replaces closes its connection 10 seconds after its session stops using it. Set `warm_standby_enabled=False` to open a connection for each reply and close it afterwards: nothing is held between replies, and every reply pays a connect. A reply that starts while the previous one is still being cancelled, and `synthesize()`, each use their own short-lived socket.

A reply with nothing to say, such as whitespace or punctuation alone, sends nothing and ends without audio. A reply that fails reaches the session as a single unrecoverable error, however many attempts and models it went through, so `AgentSession` ends a call only after several failed replies in a row, as with any other TTS. A failed attempt that the plugin retries, or that the next model in `connections=[...]` speaks, is reported as recoverable.

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

## Upgrading from 1.8.3 or earlier

These change what an existing worker does:

- `slng_base_url` has no default on STT or TTS: pass your region's host, for example `us-east.api.slng.ai`, or set `SLNG_BASE_URL`. Without either, construction raises a `ValueError`. Connections given as full endpoint URLs need neither.
- TTS `warm_standby_enabled` defaults to True, so the connection is open from session start and counts as one concurrent session for the whole call. Set it to False for a connection per reply, as before.
- TTS `text_chunking` defaults to `"sentence"`: one frame per sentence, rather than clause-sized frames. A `word_tokenizer` that is a `WordTokenizer` keeps the clause-sized frames.
- TTS no longer sends `language="en"` or `speed=1.0` when they are omitted; the model's catalog defaults apply instead.
- A TTS connection's own init (`TTSConnectionConfig(init=...)`) is merged with the plugin's settings instead of being sent as written: every field in it is kept, but the plugin's `model`, `voice`, `encoding` and `sample_rate`, and any option you set, win where both have one, so `update_options` applies to it.
- A TTS `encoding` keyword argument other than `linear16` raises a `ValueError`, where it used to reach the init and garble the audio.
