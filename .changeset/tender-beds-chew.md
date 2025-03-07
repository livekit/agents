---
"livekit-plugins-elevenlabs": patch
---

Added speed parameter for voices.

E.g.:

```python
voice = Voice(
    id="EXAVITQu4vr4xnSDxMaL",
    name="Bella",
    category="premade",
    settings=VoiceSettings(
        stability=0.71,
        speed=1.2,
        similarity_boost=0.5,
        style=0.0,
        use_speaker_boost=True,
    ),
)

```
