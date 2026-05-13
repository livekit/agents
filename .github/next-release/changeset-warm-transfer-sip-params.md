---
"livekit-agents": patch
---

Add `dtmf` and `ringing_timeout` parameters to `WarmTransferTask`. `dtmf` sends DTMF tones once the human agent answers (e.g. to dial an extension or navigate an IVR); `ringing_timeout` bounds how long to wait for an answer before falling back to the caller conversation. Both are passed through to the underlying `CreateSIPParticipantRequest`.
