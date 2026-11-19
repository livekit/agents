# Gradium plugin for LiveKit Agents

Support for [Gradium](https://gradium.ai/)-hosted models in LiveKit Agents.

More information is available in the docs for the
[TTS](https://docs.livekit.io/agents/integrations/tts/gradium/) and [STT](https://docs.livekit.io/agents/integrations/stt/gradium/) integrations.

## Installation

```bash
pip install livekit-plugins-gradium
```

## Pre-requisites

You'll need an API key from Gradium. It can be set as an environment variable: `GRADIUM_API_KEY`.

You also need to deploy a model to Gradium and will need your model endpoint to configure the plugin.

## Preemptive generation

The STT sends a `PREFLIGHT_TRANSCRIPT` with the words heard so far as soon as the server's
VAD detects a pause, before the turn is finalized. With preemptive generation (on by default), the
LLM request starts earlier and only restarts if the final transcript differs.

To turn it off and only start generation from final transcripts:

```python
from livekit.plugins import gradium

stt = gradium.STT(preflight_trigger=None)
```

### What to expect

In the best case, every word has already been transcribed when the VAD trips, and the LLM
gets its request as soon as the VAD trips instead of when the turn is finalized. The 350 ms
between the trip and the final transcript is measured against the plugin defaults
(`vad_bucket=2`, `vad_threshold=0.9`, `vad_flush=True`):

```text
ms              -480        0       300       700     1050        1500    1850
                |           |       |         |       |           |       |
user speech     ████████████                                                   user stops speaking at 0 ms
silence                     ········T·········                                 T: last words transcribed (300 ms); VAD trips at 700 ms
finalize                                      ════════                         flush, then final transcript at 1050 ms

preflight on                                  ────────────────────●            LLM request at 700 ms, first token at 1500 ms
preflight off                                         ────────────────────●    LLM request at 1050 ms, first token at 1850 ms
gain                                                              ◄───────►    first token 350 ms earlier
```

A lower `vad_bucket` or `vad_threshold` makes VAD trip earlier, bringing the VAD trip
closer to the last words, which risks triggering two LLM calls. The preview is sent again
only when new words arrive, so a VAD that flickers during a silence keeps the request already
started. When tuning the VAD parameters, also check the number of LLM
calls to ensure you still benefit from preemptive generation.
