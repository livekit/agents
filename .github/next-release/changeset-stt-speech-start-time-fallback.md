---
"livekit-agents": patch
"livekit-plugins-assemblyai": patch
---

Add `SpeechEvent.speech_start_time` so STT plugins can propagate server-provided speech onset time. The AssemblyAI plugin populates it from `SpeechStarted.timestamp`, letting the framework back-date `_speech_start_time` when local VAD doesn't fire for STT-detected speech (previously produced `speech_duration = 0.0s` exactly).
