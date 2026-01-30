# LiquidAI Audio plugin for LiveKit Agents

Support for the LFM2.5-Audio family of SST from LiquidAI.

https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF

## Installation

```bash
pip install livekit-plugins-liquidai
```

## Pre-requisites

Start audio server

```bash
export CKPT=/path/to/LFM2.5-Audio-1.5B-GGUF
./llama-liquid-audio-server -m $CKPT/LFM2.5-Audio-1.5B-Q4_0.gguf -mm $CKPT/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf -mv $CKPT/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf --tts-speaker-file $CKPT/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf
```