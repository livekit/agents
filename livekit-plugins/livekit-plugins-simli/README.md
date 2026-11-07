# Simli plugin for LiveKit Agents

Support for the [Simli](https://simli.com/) virtual avatar.

See the [Simli integration docs](https://docs.livekit.io/agents/models/avatar/plugins/simli/) for more information.

## Installation

```bash
pip install livekit-plugins-simli
```

## Pre-requisites

You'll need an API key and a face ID from Simli. The plugin doesn't read either from the environment, so pass them to `SimliConfig`, for example from an environment variable such as `SIMLI_API_KEY`:

```python
import os

from livekit.plugins import simli

avatar = simli.AvatarSession(
    simli_config=simli.SimliConfig(
        api_key=os.environ["SIMLI_API_KEY"],
        face_id="<your-face-id>",
    ),
)
```
