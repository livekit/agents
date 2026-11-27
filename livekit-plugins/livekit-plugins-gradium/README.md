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

The STT ends a turn when the server's VAD detects a pause: it asks the server to flush the words
the model still holds, and emits the final transcript on the server's reply. Before that, as soon
as the VAD trips, it sends a `PREFLIGHT_TRANSCRIPT` with the words heard so far. With preemptive generation (on by default), the
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
(`vad_bucket=2`, `vad_threshold=0.9`):

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

A lower `vad_bucket` or `vad_threshold` makes the VAD trip earlier, closer to the last words.
That ends turns sooner, but a pause in the middle of a sentence is more likely to end the turn
early, and a word still held by the model at the trip makes the final transcript differ from the
preview, which costs a second LLM call. When tuning the VAD parameters, also check the number of
LLM calls to ensure you still benefit from preemptive generation.

### Pauses inside a sentence

By default the first VAD trip ends the turn once the server answers the flush. If the user pauses
mid-sentence and resumes, the sentence is split into two turns and the first LLM request is
wasted. `vad_confirm_steps` trades latency for fewer of those early endings: after the trip, the
turn ends only when the server has answered the flush **and** that many more steps (80 ms each)
had the VAD above `vad_threshold`.

```python
stt = gradium.STT(vad_confirm_steps=3)
```

Steps below the threshold don't count, but don't restart the count either: after the user stops,
the VAD can hover around the threshold for seconds, and that must not hold the turn open. What
keeps the turn open is new words after the flush reply: the user is talking again, the count
restarts, and the next trip flushes again, so words the model held in the meantime are not lost.
The preview is only sent again if the words changed.

The trade-off:

- `0` (default): lowest latency, mid-sentence pauses end the turn.
- `N > 0`: the turn ends once the flush reply has arrived and `N` steps above the threshold have
  followed the trip, so at least `N × 80 ms` after the trip. A pause the user resumes from costs
  an extra flush request, and a second preview if the words changed.
- A sound the model does not transcribe (a breath, an "mm") while the VAD stays above the
  threshold does not keep the turn open.
