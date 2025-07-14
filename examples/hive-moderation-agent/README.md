# LiveKit realtime moderation agent using Hive

This is an agent that performs visual moderation of every participant's video in a room.  It does this moderation using the Visual Content Moderation model from [Hive](https://thehive.ai) [[docs](https://docs.thehive.ai/docs/visual-content-moderation#visual-content-moderation)].

## Prerequisites

Before running this agent, you'll need:

1. A LiveKit Cloud project (or a self-hosted LiveKit server).
2. An API key from Hive to access the above mentioned model.

## Configuration

Currently, this agent is configured entirely from the `agent.py` source code and the environment.

### Environment Variables

| configuration | description | example value |
|---------------|-------------|---------------|
| `LIVEKIT_URL` | Your LiveKit URL | `wss://test-abc123de.livekit.cloud` |
| `LIVEKIT_API_KEY` | Your LiveKit API key | |
| `LIVEKIT_API_SECRET` | Your LiveKit API secret | |
| `HIVE_API_KEY` | The API key from Hive to access the `Visual Content Moderation` model | `abc1deFgHIjK23KLMNOp45QrsTuv6wx8` |

### Code

| configuration | description | example value |
|---------------|-------------|---------------|
| `MOD_FRAME_INTERVAL` | Minimum number of seconds to wait between frames | 5.0 |
| `HIVE_HEADERS` | The headers to send with every request to the Hive API | `{}` |
| `CONFIDENCE_THRESHOLD` | The minimum score Hive's moderation class must meet before it is considered a problem | 0.9 |

## Running

Run this code like you would any other [LiveKit agent](https://docs.livekit.io/agents/build/anatomy/#starting-the-worker):

```
python3 agent.py start
```

Once running, the agent will join all new LiveKit rooms by default and begin moderation.
