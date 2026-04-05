# D-ID plugin for LiveKit Agents

Support for the [D-ID](https://d-id.com/) virtual avatar.

See the [D-ID integration docs](https://docs.livekit.io/agents/models/avatar/plugins/did/) for more information.

## Installation

```bash
pip install livekit-plugins-did
```

## Pre-requisites

You'll need an API key from D-ID. It can be set as an environment variable: `DID_API_KEY`

## Supported avatars

This plugin only supports **v4 avatars** (type: `expressive`). Earlier avatar versions are not compatible. See the [D-ID Create Agent API](https://docs.d-id.com/reference/createagent) for details on creating a compatible agent.

Example — creating an expressive agent via the D-ID API:
```bash
curl -X POST https://api.d-id.com/agents \
  -H "Authorization: Basic <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "presenter": {
      "type": "expressive",
      "presenter_id": "public_mia_elegant@avt_TJ0Tq5"
    },
    "preview_name": "My Expressive Agent"
  }'
```

Use the agent ID from the response as the `agent_id` parameter in the plugin.
