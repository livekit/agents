# Vosk Plugin for LiveKit Agents

Offline speech-to-text plugin using [Vosk](https://alphacephei.com/vosk/) for the LiveKit Agents framework.

## Installation

```bash
pip install livekit-plugins-vosk
```

## Pre-requisites

Download a Vosk model and set its path via `model_path` or the `VOSK_MODEL_PATH` environment
variable.

## Logging

By default, the Vosk native logger is silenced. Pass `log_level=0` to `vosk.STT(...)` to
enable Vosk logs, or `log_level=None` to keep whatever global Vosk log level is already set.
