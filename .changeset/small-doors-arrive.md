---
"livekit-agents": patch
"livekit-plugins-openai": patch
---

Support OpenAI Assistants API as a beta feature under `livekit.plugins.openai.beta`
Add _metadata to ChatCtx and ChatMessage which can be used (in the case of OpenAI assistants) for bookeeping to sync local state with remote, OpenAI state
