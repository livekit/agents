---
"livekit-agents": patch
---

Add `inference.LLM.update_options(model=..., extra_kwargs=...)` so the model and per-request options can be swapped live without recreating the LLM. Symmetric with the existing `inference.STT.update_options` / `inference.TTS.update_options`.
