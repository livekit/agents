---
"livekit-agents": patch
"livekit-plugins-xai": patch
---

Honor `AgentSession.say(..., allow_interruptions=False)` with xAI server turn detection by sending `interruptible: false` on native `force_message` items.
