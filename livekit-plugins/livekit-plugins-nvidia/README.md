<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NVIDIA plugin for LiveKit Agents

Support for NVIDIA Speech AI services in LiveKit Agents.

## Installation

```bash
pip install livekit-plugins-nvidia
```

## Pre-requisites

You can either:

1. Use an API key from NVIDIA. It can be set as an environment variable: `NVIDIA_API_KEY`
2. Use your self-hosted [NIM](https://developer.nvidia.com/nim) server.

## Usage

```python
from livekit.plugins import nvidia

stt = nvidia.STT(
    model="parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer",
    inference_mode="streaming",
    server="grpc.nvcf.nvidia.com:443",
    use_ssl=True,
    endpointing=nvidia.EndpointingConfig(mode="low_latency"),
)

tts = nvidia.TTS(
    voice="Magpie-Multilingual.EN-US.Leo",
    sample_rate=16000,
    inference_mode="online",
)
```

STT defaults to `inference_mode="auto"`; use `"streaming"` or `"offline"` to
match the deployed model. TTS defaults to `"online"`; use `"offline"` for batch
models.

For local NVIDIA Speech deployments, pass the local `server` and set
`use_ssl=False` when TLS is not enabled.
