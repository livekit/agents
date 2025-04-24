# DataStream Audio Example

This example demonstrates how to use LiveKit's DataStream feature to send and receive audio between agents.

## Overview

The example consists of two main components:

1. **Audio Receiver (`audio_receiver.py`)**: Receives audio from a sender, streams it to a LiveKit room, and handles interruptions and playback notifications.

2. **Agent Worker (`agent_worker.py`)**: Implements an agent with audio capabilities that can connect to a LiveKit room.


### Starting the Agent Worker

```bash
python examples/other/datastream-audio/agent_worker.py dev
```

### Starting the Audio Receiver

```bash
python examples/other/datastream-audio/audio_receiver.py <room-name>
```

Replace `<room-name>` with the name of the LiveKit room you want to connect to.


## How It Works

1. The agent worker connects to the room and output audio to a datastream.
2. The audio receiver connects to the same room and waits for an audio sender to join.
3. When audio is received via DataStream, it's forwarded to the room through an audio track.
4. The receiver handles interruptions and notifies the sender when playback is complete.

