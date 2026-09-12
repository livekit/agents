# Vui plugin for LiveKit Agents

Support for [Vui Nano](https://github.com/fluxions-ai/vui) text-to-speech in LiveKit Agents — a small, context-aware TTS model trained on real conversations (219M active parameters, 305M total, Apache 2.0) that runs **in-process**: on an NVIDIA GPU, or on MLX on Apple Silicon. No API key, nothing leaves the machine.

Streaming input is supported: sentences are rendered as the LLM produces them, in one conversation row, so prosody carries across the reply.

## Installation

```bash
pip install livekit-plugins-vui
```

Python 3.12. Weights (`vui-nano-1.1`, ~1.2 GB) and the shipped voice prompts download from Hugging Face on first use; call `tts.prewarm()` to do that ahead of the first request.

On CUDA, install a [flash-attn](https://github.com/Dao-AILab/flash-attention) wheel matching your torch/CUDA for full speed (~10× realtime on a 5090); without it the engine uses PyTorch's SDPA fallback — same output, ~2.7× realtime. Apple Silicon uses MLX and needs nothing extra.

## Usage

```python
from livekit.plugins import vui

tts = vui.TTS(voice="maeve")  # maeve / abraham / rhian / harry, a prompt .safetensors, or a .wav to clone
```

Pass `checkpoint="vui-190k"` (or a local path) to run a different checkpoint, and `temperature=` to change sampling.
