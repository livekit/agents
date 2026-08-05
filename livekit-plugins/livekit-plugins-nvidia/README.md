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

STT defaults to `inference_mode="auto"` for backward compatibility. Set it to
`"streaming"` or `"offline"` when the deployed model supports only one API so
LiveKit receives exact capabilities. For offline TTS models, set the TTS
`inference_mode="offline"`; streaming TTS models use the default `"online"` mode.

For local NVIDIA Speech deployments, pass the local `server` and set
`use_ssl=False` when TLS is not enabled.

The STT and TTS constructors keep common voice-agent options flat.
Provider-specific endpointing is grouped under `EndpointingConfig`, and
lower-level provider settings can be passed through the `options` escape hatch.

The plugin supports `nvidia-riva-client` versions from 2.16 through 2.26.x.
Zero-shot prompt and quality arguments are translated to the names used by the
installed client, while unsupported options fail before a synthesis request is sent.

LiveKit `SynthesizeStream` instances represent one text segment. Create a new stream
after calling `flush()` instead of pushing another segment into the same stream.

The experimental PersonaPlex realtime model is audio-in/audio-out only. Manual
`generate_reply()` calls are not supported yet.
