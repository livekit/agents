# LiveKit Plugins HuggingFace

Agent Framework plugin for local HuggingFace model inference with optional TurboQuant-GPU KV cache compression.

## Installation

```bash
pip install livekit-plugins-hf
```

With TurboQuant-GPU support:

```bash
pip install livekit-plugins-hf[turboquant]
```

## Usage

```python
from livekit.plugins.hf import LLM

# Basic local inference
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# With TurboQuant-GPU KV cache compression
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    turboquant=True,
    turboquant_bits=3,
)
```
