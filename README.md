<!--BEGIN_BANNER_IMAGE-->
<!--END_BANNER_IMAGE-->

# LiveKit Agent Framework

The Agent Framework is designed for building real-time, programmable participants
that run on servers. Easily tap into LiveKit WebRTC sessions and process or generate
audio, video, and data streams.

The framework includes plugins for common workflows, such as voice activity detection and speech-to-text.

Furthermore, it integrates seamlessly with LiveKit server, offloading job queuing and scheduling responsibilities to it. This approach eliminates the need for additional queuing infrastructure. The code developed on your local machine is fully scalable when deployed to a server, supporting thousands of concurrent sessions.

## Getting Started

To install the core agent library:

```bash
pip install livekit-agents
```

Plugins can be installed individually depending on what your agent needs. Available plugins:

- livekit-plugins-elevenlabs
- livekit-plugins-openai
- livekit-plugins-silero
- livekit-plugins-anthropic

## Terminology

- **Agent**: A function that defines the workflow of the server-side participant. This is what you will be developing.
- **Worker**: A container process responsible for managing job queuing with LiveKit server. Each worker is capable of running multiple agents simultaneously.
- **Plugin**: A library class that perform a specific task like speech-to-text with a specific provider. Agents can combine multiple plugins together to perform more complex tasks.

## Creating an Agent

Let's begin with a simple agent that performs speech-to-text on incoming audio tracks and sends a data channel message for each result.

```python title="my_agent.py"
import asyncio
import json
import logging
from typing import Optional, Set
from livekit import agents, rtc
from livekit.plugins.vad import VADPlugin, VADEventType
from livekit.plugins.openai import WhisperAPITranscriber


class MyAgent():
    def __init__(self):
        # Initialize plugins 
        self.vad_plugin = VADPlugin(
            left_padding_ms=1000,
            silence_threshold_ms=500)
        self.stt_plugin = WhisperAPITranscriber()

        self.ctx: Optional[agents.JobContext] = None
        self.track_tasks: Set[asyncio.Task] = set()
        self.ctx = None

    async def start(self, ctx: agents.JobContext):
        self.ctx = ctx
        ctx.room.on("track_subscribed", self.on_track_subscribed)
        ctx.room.on("disconnected", self.cleanup)

    # Callback for when a track is subscribed to. Only tracks matching the should_subscribe filter that is configured when accepting a job will be subscribed to.
    def on_track_subscribed(
            self,
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant):
        t = asyncio.create_task(self.process_track(track))
        self.track_tasks.add(t)
        t.add_done_callback(self.track_tasks.discard)

    def cleanup(self):
        # Whatever cleanup you need to do.
        pass

    async def process_track(self, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        async for vad_result in self.vad_plugin.start(audio_stream):
            # When the human has finished talking, send the audio frames containing voice to the transcription plugin (in this case the Whisper API).
            if vad_result.type == VADEventType.FINISHED:
                stt_output = await self.stt_plugin.transcribe_frames(vad_result.frames)
                if len(stt_output) == 0:
                    continue
                # Send the speech transcription to all participants in the LiveKit room via a DataChannel message.
                await self.ctx.room.local_participant.publish_data(json.dumps({"type": "transcription", "text": text}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Callback that gets called on every new Agent JobRequest. In this callback you can create your agent and accept (or decline) a job. Declining a job will tell the LiveKit server to give the job to another Worker.
    async def job_request_cb(job_request: agents.JobRequest):
        # Accept the job request with my_agent and configure a filter function that decides which tracks the agent processes. In this case, the agent only cares about audio tracks.
        my_agent = MyAgent()
        await job_request.accept(my_agent.start, should_subscribe=lambda track_pub, _: track_pub.kind == rtc.TrackKind.KIND_AUDIO)

    # When a new LiveKit room is created, the job_request_cb is called.
    worker = agents.Worker(job_request_cb=job_request_cb,
                           worker_type=agents.JobType.JT_ROOM)

    # Start the cli
    agents.run_app(worker)
```

## Running an Agent

The Agent Framework expose a cli interface to run your agent. To start the above agent, run:

```bash
python my_agent.py start --api-key=<your livekit api key> --api-secret<your livekit api secret> --url=<your livekit url>
```

The environment variables `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and `LIVEKIT_URL` can be used instead of the cli-args. (`.env` files also supported)

### What is happening when I run my Agent?

When you run your agent with the above commands, a worker is started that opens an authenticated websocket connection to a LiveKit server (defined by your LIVEKIT_URL and authenticated with the key and secret).

This doesn't actually run any agents. Instead, the worker sits waiting for the LiveKit server to give it a job. In the above case, this happens whenever a new LiveKit room is created because the worker type is `JT_ROOM`.

Once the LiveKit server is given a job, the worker can decide whether or not to accept it. Accepting the job will create a LiveKit participant that joins the room and begin subscribing to tracks.

### What happens when I SIGTERM one of my Workers? 

The Agent Framework was designed for production use cases. Since agents are more stateful entities than typical web-servers, it's important that workers can't be terminated while they are running active agents.

When calling SIGTERM on a worker, the worker will signal to the LiveKit server that it does not want to be given any more jobs. It will also auto-decline any new job requests that might sneak in before the server signaling has occurred. The worker will remain alive while it manages an agent that is still connected to a room.

## Deploying a Worker?

Workers can be deployed like any other python application. Deployments will typically need `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and `LIVEKIT_URL` set as environment variables.

This reference [Dockerfile](examples/agents/Dockerfile) serves as a good start point for your agent deployment.

## More Examples

Examples can be found in the `examples/` repo.

Examples coming soon:

- KITT (Clone of ChatGPT Voice Mode) 
- Audio-to-Audio Language Translation
- Transcription
- Face-Detection
- Voice-to-Image

<!--BEGIN_REPO_NAV-->
<!--END_REPO_NAV-->
