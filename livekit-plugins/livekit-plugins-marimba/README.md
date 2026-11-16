# Marimba plugin for LiveKit Agents

Audio-only end-of-turn detection using [Marimba](https://huggingface.co/charcot-ai/marimba).
The detector scores whether the speaker has finished their turn without using
a transcript. It accepts all language codes.

Paper: [marimba: A System One Model for Speech](https://arxiv.org/abs/2611.18330).

## Installation

```sh
pip install livekit-plugins-marimba
pip install "marimba-audio[torch]"
```

The plugin depends on the base `marimba-audio` package, not PyTorch. Install a
Marimba inference backend separately as shown above. Model files are downloaded
on first use.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import marimba, silero

session = AgentSession(
    vad=silero.VAD.load(min_silence_duration=0.25),
    turn_handling={"turn_detection": marimba.TurnDetector()},
    # Add your STT, LLM, and TTS providers here.
)
```

Install `livekit-plugins-silero` separately for this VAD. The detector starts
loading the model in a background thread when its first stream is opened.
To exclude initialization from the first turn, preload a model using
`from marimba import Marimba` from the `marimba-audio` package and pass it as
`TurnDetector(model=model)`.

`unlikely_threshold` defaults to 0.5. Below this probability LiveKit waits for
the maximum endpointing delay. `prediction_timeout` defaults to 2 seconds and
must be greater than zero. CPU latency depends on hardware; no real-time
guarantee is claimed.

Audio is converted to mono, resampled to 16 kHz with soxr HQ, and limited to the
latest eight seconds. Inference runs off the event loop. Failed or empty
predictions return probability 1.0, matching LiveKit's local fail-open behavior.
Cancellation resolves the pending prediction with 0.0 and ignores stale results.
Flushing clears the retained audio.
