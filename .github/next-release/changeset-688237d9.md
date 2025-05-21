---
"livekit-plugins-turn-detector": patch
---

multilingual model update (#2219)
- broad accuracy improvements
- more robust to STT punctuation errors
- better quantization process to preserve accuracy
- better threshold calibration and evaluation metrics using the quantized model instead of full pytorch model
