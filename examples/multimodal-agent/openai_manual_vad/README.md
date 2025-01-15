# Push to Talk Example

This example demonstrates how to manually control the VAD of the OpenAI realtime agent using LiveKit's [RPC functionality](https://docs.livekit.io/home/client/data/rpc/).

## How It Works

1. The agent sets a `supports-ptt` attribute to indicate it supports push-to-talk functionality
2. The agent registers two RPC methods: `ptt.start` and `ptt.end` to handle push/release actions
3. When the button is pressed, the frontend sends an RPC call to `ptt.start` to interrupt the agent
4. When the button is released, the frontend sends an RPC call to `ptt.end` to generate a reply

## Frontend Integration

Here's a basic example of how to implement PTT in your frontend using LiveKit's RPC:

```javascript
// Find agent participant that supports PTT
const agent = participants.find(
  (p) => p.attributes?.["supports-ptt"] === "1"
);
if (!agent) return;

// Handle push to talk start
const handlePushStart = async () => {
  try {
    await localParticipant.setMicrophoneEnabled(true);
    await localParticipant.performRpc({
      destinationIdentity: agent.identity,
      method: "ptt.start",
    });
  } catch (error) {
    console.error("Failed to send PTT push:", error);
  }
};

// Handle push to talk end
const handlePushEnd = async () => {
  try {
    await localParticipant.setMicrophoneEnabled(false);
    await localParticipant.performRpc({
      destinationIdentity: agent.identity,
      method: "ptt.end",
    });
  } catch (error) {
    console.error("Failed to send PTT release:", error);
  }
};
```

Note: In a production environment, consider implementing PTT heartbeats to handle potential state synchronization issues between the frontend and agent.

For more details about LiveKit's RPC functionality, see the [RPC documentation](https://docs.livekit.io/home/client/data/rpc/).

## Running the Example

1. Start the agent:
   ```bash
   python push_to_talk.py dev
   ```

2. Implement the frontend RPC calls as shown in the example above
