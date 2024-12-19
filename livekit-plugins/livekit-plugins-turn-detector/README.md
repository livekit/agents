# LiveKit Plugins Turn Detector

This plugin introduces end-of-turn detection for LiveKit Agents using a custom open-weight model to determine when a user has finished speaking.

Traditional voice agents use VAD (voice activity detection) for end-of-turn detection. However, VAD models lack language understanding, often causing false positives where the agent interrupts the user before they finish speaking.

By leveraging a language model specifically trained for this task, this plugin offers a more accurate and robust method for detecting end-of-turns. The current version supports English only and should not be used when targeting other languages.

## Installation

```bash
pip install livekit-plugins-turn-detector
```

## Usage

This plugin is designed to be used with the `VoicePipelineAgent`:

```python
from livekit.plugins import turn_detector

agent = VoicePipelineAgent(
    ...
    turn_detector=turn_detector.EOUModel(),
)
```

## Running your agent

This plugin requires model files. Before starting your agent for the first time, or when building Docker images for deployment, run the following command to download the model files:

```bash
python my_agent.py download-files
```

## Model system requirements

The end-of-turn model is optimized to run on CPUs with modest system requirements. It is designed to run on the same server hosting your agents. On a 4-core server instance, it completes inference in ~50ms with minimal CPU usage.

The model requires 1.5GB of RAM and runs within a shared inference server, supporting multiple concurrent sessions.

We are working to reduce the CPU and memory requirements in future releases.

## License

The plugin source code is licensed under the Apache-2.0 license.

The end-of-turn model is licensed under the [LiveKit Model License](https://huggingface.co/livekit/turn-detector/blob/main/LICENSE).
