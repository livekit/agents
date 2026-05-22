---
"livekit-plugins-rime": patch
---

Add the `time_scale_factor` parameter to the Rime TTS plugin for the `arcana`, `mistv3`, and `coda` models. Values above 1.0 slow down the audio; values below 1.0 speed it up. Attempting to set `time_scale_factor` on `mistv2` raises `ValueError`.
