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
    language="en",
)
```

Additional keyword arguments are forwarded to the gateway and applied according to the selected model's contract. Failover across multiple models or endpoints is available via `connections=[...]`; see [docs.slng.ai](https://docs.slng.ai/) for details.

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
- TTS `voice` is required and passed verbatim as the provider's voice identifier.
- Language codes are no longer normalized client-side; send the value the model expects (for example BCP-47 `hi-IN` for Sarvam, not `hi`).
- STT `recognize()` (HTTP batch) is no longer supported; use `stream()`. Only `pcm_s16le` input audio is supported.
- `api_token` still works on STT but is deprecated; use `api_key`.
