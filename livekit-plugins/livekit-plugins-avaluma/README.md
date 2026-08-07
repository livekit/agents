# Avaluma plugin for LiveKit Agents

Support for the [Avaluma](https://avaluma.ai/) virtual avatar.

See the [Avaluma integration docs](https://docs.livekit.io/agents/models/avatar/plugins/avaluma/) for more information.

## Installation

```bash
pip install livekit-plugins-avaluma
```

## Pre-requisites

You'll need a license key and an avatar ID from Avaluma. They can be set as environment variables: `AVALUMA_LICENSE_KEY`, `AVALUMA_AVATAR_ID`

`AVALUMA_AVATAR_SERVER_URL` is optional. It defaults to Avaluma's hosted service and only needs to be set when you self-host the avatar server.
