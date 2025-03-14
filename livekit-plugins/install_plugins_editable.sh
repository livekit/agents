#!/bin/bash
set -e

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "You are not in a virtual environment."
    exit 1
fi
pip install -e ./livekit-plugins-anthropic --config-settings editable_mode=strict
pip install -e ./livekit-plugins-aws --config-settings editable_mode=strict
pip install -e ./livekit-plugins-assemblyai --config-settings editable_mode=strict
pip install -e ./livekit-plugins-azure --config-settings editable_mode=strict
pip install -e ./livekit-plugins-cartesia --config-settings editable_mode=strict
pip install -e ./livekit-plugins-deepgram --config-settings editable_mode=strict
pip install -e ./livekit-plugins-elevenlabs --config-settings editable_mode=strict
pip install -e ./livekit-plugins-fal --config-settings editable_mode=strict
pip install -e ./livekit-plugins-google --config-settings editable_mode=strict
pip install -e ./livekit-plugins-minimal --config-settings editable_mode=strict
pip install -e ./livekit-plugins-nltk --config-settings editable_mode=strict
pip install -e ./livekit-plugins-openai --config-settings editable_mode=strict
pip install -e ./livekit-plugins-rag --config-settings editable_mode=strict
pip install -e ./livekit-plugins-rime --config-settings editable_mode=strict
pip install -e ./livekit-plugins-llama-index --config-settings editable_mode=strict
pip install -e ./livekit-plugins-turn-detector --config-settings editable_mode=strict
pip install -e ./livekit-plugins-silero --config-settings editable_mode=strict
pip install -e ./livekit-plugins-speechmatics --config-settings editable_mode=strict
pip install -e ./livekit-plugins-neuphonic --config-settings editable_mode=strict
pip install -e ./livekit-plugins-browser --config-settings editable_mode=strict

