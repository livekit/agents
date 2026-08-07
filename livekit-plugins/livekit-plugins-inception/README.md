# Inception AI plugin for LiveKit Agents

Support for Inception AI LLM models (e.g. `mercury-2` diffusion model) with LiveKit.

## Installation

```bash
pip install livekit-plugins-inception
```

## Pre-requisites

You'll need an API key from Inception AI. It can be set as an environment variable: `INCEPTION_API_KEY`

## Usage

### LLM

```python
from livekit.plugins import inception

llm = inception.LLM(model="mercury-2")
```
