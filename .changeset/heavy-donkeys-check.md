---
"livekit-agents": patch
---

Fix bug where if the tts_source was a string but before_tts_cb returned AsyncIterable[str], the transcript would not be synthesized.