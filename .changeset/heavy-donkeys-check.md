---
"livekit-agents": patch
---

Fix bug in before_tts_cb where AsyncIterable[str] was not being handled correctly.
