# Push to Talk Example

This example demonstrates how to manually control the VAD of the OpenAI realtime agent using LiveKit's [RPC functionality](https://docs.livekit.io/home/client/data/rpc/).

## How It Works

1. The agent sets a `supports-ptt` attribute to indicate it supports push-to-talk functionality
2. The agent registers an RPC method `ptt` that handles push/release actions
3. When the button is pressed, the frontend sends an RPC call with `push` payload to interrupt the agent
4. When the button is released, the frontend sends an RPC call with `release` payload to commit the audio buffer

## Frontend Integration

A complete frontend implementation can be found in the [voice-assistant-frontend](https://github.com/livekit-examples/voice-assistant-frontend) repository. The frontend will:

1. Check for the `supports-ptt` attribute on the agent
2. If PTT is supported, enable the push-to-talk button
3. Send RPC calls to the agent when the button is pressed/released

## Running the Example

1. Start the agent:
   ```bash
   python push_to_talk.py dev
   ```

2. Run the frontend application from [voice-assistant-frontend](https://github.com/livekit-examples/voice-assistant-frontend)
