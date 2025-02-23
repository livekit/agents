# LiveKit Avatar Example

This example demonstrates how to create an animated avatar that responds to audio input using LiveKit's agent system. The avatar worker generates synchronized video based on received audio input.

## How it Works

1. The agent sends connection info (including token, room name, and URL) to the avatar dispatcher server
2. The dispatcher launches an avatar worker process for that room
3. The agent streams audio to the avatar worker using LiveKit's DataStream
4. The avatar worker:
   - Receives the audio stream
   - Generates synchronized video frames based on the audio
   - Publishes both the audio and video back to the room

## Usage

1. Start the avatar dispatcher server:
```bash
python examples/avatar/simli_avatar/dispatcher.py [--port 8089]
```

2. Start the agent worker:
```bash
python examples/avatar/simli_avatar/agent_worker.py dev [--avatar-url http://localhost:8089/launch]
```
