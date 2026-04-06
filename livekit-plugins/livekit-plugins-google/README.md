# Google AI plugin for LiveKit Agents

Support for Gemini, Gemini Live, Cloud Speech-to-Text, and Cloud Text-to-Speech.

See [https://docs.livekit.io/agents/integrations/google/](https://docs.livekit.io/agents/integrations/google/) for more information.

## Installation

```bash
pip install livekit-plugins-google
```

## Pre-requisites

For credentials, you'll need a Google Cloud account and obtain the correct credentials. Credentials can be passed directly or via Application Default Credentials as specified in [How Application Default Credentials works](https://cloud.google.com/docs/authentication/application-default-credentials).

To use the STT and TTS API, you'll need to enable the respective services for your Google Cloud project.

- Cloud Speech-to-Text API
- Cloud Text-to-Speech API

## GCS Egress

When starting a LiveKit Egress job that writes to Google Cloud Storage, use
`build_gcp_upload` to construct the `GCPUpload` configuration from a service account key:

```python
import json
from livekit import api
from livekit.plugins.google import build_gcp_upload
from livekit.protocol.egress import RoomCompositeEgressRequest, EncodedFileOutput

async with api.LiveKitAPI() as lkapi:
    await lkapi.egress.start_room_composite_egress(
        RoomCompositeEgressRequest(
            room_name="my-room",
            file_outputs=[EncodedFileOutput(
                gcp=build_gcp_upload("my-bucket", credentials_file="/path/to/sa-key.json"),
                filepath="recordings/{room_name}/{time}.mp4",
            )],
        )
    )
```

`build_gcp_upload` resolves credentials in this order:

1. `credentials_info` — a service account key dict passed directly
2. `credentials_file` — a path to a service account JSON key file
3. `GOOGLE_APPLICATION_CREDENTIALS` environment variable pointing to a key file

### GKE Workload Identity

When your **agent** pod runs on GKE with Workload Identity, it does not have a service
account key file — authentication is handled transparently via short-lived tokens from the
GKE metadata server.  These tokens cannot be embedded in `GCPUpload.credentials`.

The recommended approach for GKE deployments is to run a **self-hosted LiveKit Egress
Server** whose pod has a Workload Identity binding with GCS write permissions, and pass an
empty `credentials` field so that the Egress Server authenticates using its own ambient
credentials:

```python
from livekit.api import GCPUpload
from livekit.protocol.egress import RoomCompositeEgressRequest, EncodedFileOutput

# The Egress Server pod must have GCS write permissions via its own Workload Identity.
await lkapi.egress.start_room_composite_egress(
    RoomCompositeEgressRequest(
        room_name="my-room",
        file_outputs=[EncodedFileOutput(
            gcp=GCPUpload(bucket="my-bucket"),  # credentials="" → Egress Server uses its own ADC
            filepath="recordings/{room_name}/{time}.mp4",
        )],
    )
)
```

See the [GKE Workload Identity documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity)
for how to bind a Kubernetes service account to a GCP service account and grant it GCS
permissions (`roles/storage.objectCreator` on the target bucket).

## Live API model support

LiveKit supports both Gemini Live API on both Gemini Developer API as well as Vertex AI. However, be aware they have slightly different behavior and use different model names.

The following models are supported by Gemini Developer API:

- gemini-2.5-flash-native-audio-preview-09-2025
- gemini-2.5-flash-native-audio-preview-12-2025

And these on Vertex AI:

- gemini-live-2.5-flash-native-audio

References:

- [Gemini API Models](https://ai.google.dev/gemini-api/docs/models)
- [Vertex Live API](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api)
