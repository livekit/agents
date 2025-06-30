---
"livekit-plugins-sarvam": patch
---

handling multiple audio chunk output (#2641)
Ensuring all returned audio segments from Sarvam TTS are decoded and pushed, so long sentences are no longer truncated.
