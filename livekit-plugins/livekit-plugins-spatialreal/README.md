# SpatialReal plugin for LiveKit Agents

Support for the [SpatialReal](https://spatialreal.com/) virtual avatar.

The agent's speech is sent to SpatialReal, which joins the room and publishes the avatar's audio and
animation; the SpatialReal client SDK renders the avatar on the viewer's own device.

## Installation

```bash
pip install livekit-plugins-spatialreal
```

## Pre-requisites

You'll need an API key, an app ID and an avatar ID from SpatialReal. They can be set as environment
variables: `SPATIALREAL_API_KEY`, `SPATIALREAL_APP_ID`, `SPATIALREAL_AVATAR_ID`.

```python
from livekit.plugins import spatialreal

avatar = spatialreal.AvatarSession()
await avatar.start(session, room=ctx.room)
```

## Endpoints

The SDK resolves them from the deployment it is told to use, so there is nothing to configure for
the public one. Name another with `environment=` (or `SPATIALREAL_ENVIRONMENT`), and give explicit
URLs for a deployment SpatialReal hands you an address for:

```python
avatar = spatialreal.AvatarSession(
    console_endpoint_url="https://api.private.example",
    ingress_endpoint_url="wss://driven.private.example/v2/driveningress",
)
```

`SPATIALREAL_CONSOLE_ENDPOINT` and `SPATIALREAL_INGRESS_ENDPOINT` do the same from the environment.
