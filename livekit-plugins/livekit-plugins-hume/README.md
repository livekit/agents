# Hume AI TTS plugin for LiveKit Agents

Support for text-to-speech with [Hume](https://www.hume.ai/).

See [https://docs.livekit.io/agents/integrations/tts/hume/](https://docs.livekit.io/agents/integrations/tts/hume/) for more information.

## Installation

```bash
pip install livekit-plugins-hume
```

You will need an API Key from Hume, it can be set as an environment variable: `HUME_API_KEY`. You can get it from [here](https://platform.hume.ai/settings/keys)

## Audio decisions

`DecisionModel` calls Hume's [Audio Decisions API](https://dev.hume.ai/audio-decisions/docs/overview), which answers decisions by listening to a recording instead of reading a transcript. Use it for things a transcript can't show, such as tone, frustration, naturalness or whether one speaker talks over another. It is an `llm.DecisionModel` with the `audio_input` capability, so it takes the same `Probability`, `Choice` and `Score` decisions and returns a `DecisionResponse`.

```python
from livekit.agents import llm
from livekit.plugins import hume

model = hume.DecisionModel()
response = await model.evaluate(
    audio=frames,  # 5 s to 5 min of rtc.AudioFrame
    input="A support call between a voice AI agent and a caller.",  # optional context
    decisions={
        "frustrated": llm.Probability("Does the caller sound frustrated?"),
        "naturalness": llm.Score(
            "How natural does the agent sound?",
            levels=["Not at all", "2", "3", "4", "Extremely"],
        ),
    },
)
print(response.results["frustrated"].value)  # share of raters who would answer yes
```

Every call needs `audio`. `input` is sent as Hume's `context` (up to 1,000 characters) and describes the recording. Frames are mixed to mono and resampled to 16 kHz before upload, which is what the model hears. Each answer estimates the share of human raters who would give it, so set thresholds from audio you have labeled. `response.request_id` is Hume's run id.
