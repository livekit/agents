---
"livekit-plugins-neuphonic": patch
---

Drop redundant API key from WebSocket URL query string (it was already being sent via the X-API-KEY header).
