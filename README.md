<!--BEGIN_BANNER_IMAGE-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/.github/banner_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="/.github/banner_light.png">
  <img style="width:100%;" alt="The LiveKit icon, the name of the repository and some sample code in the background." src="https://raw.githubusercontent.com/livekit/agents/main/.github/banner_light.png">
</picture>

<!--END_BANNER_IMAGE-->

# LiveKit Agent Framework

<!--BEGIN_DESCRIPTION-->

The Agent Framework is designed for building real-time, programmable participants
that run on servers. Easily tap into LiveKit WebRTC sessions and process or generate
audio, video, and data streams.

<!--END_DESCRIPTION-->

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
- livekit-plugins-google
- livekit-plugins-deepgram
- livekit-plugins-nltk

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Callback that gets called on every new Agent JobRequest. In this callback you can create your agent and accept (or decline) a job. Declining a job will tell the LiveKit server to give the job to another Worker.
    async def job_request_handler(job_request: agents.JobRequest):
        def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
            # do work on the track
            pass

        def on_disconnected():
            # do your agent's cleanup
            pass
            
        async def my_agent(self, ctx: agents.JobContext):
            ctx.room.on("track_subscribed", on_track_subscribed)
            ctx.room.on("disconnected", on_disconnected)

        # Accept the job request with my_agent and configure a filter function that decides which tracks the agent processes. In this case, the agent only cares about audio tracks.
        await job_request.accept(
            my_agent,
            identity="agent",
            subscribe_cb=agents.AutoSubscribe.AUDIO_ONLY,
            auto_disconnect_cb=agents.AutoDisconnect.DEFAULT,
        )

    # When a new LiveKit room is created, the request_handler is called.
    worker = agents.Worker(request_handler=job_request_handler,
                           worker_type=agents.JobType.JT_ROOM)

    # Start the cli
    agents.run_app(worker)
```

## Running an Agent

The Agent Framework expose a cli interface to run your agent. To start the above agent, run:

```bash
python my_agent.py start --api-key=<your livekit api key> --api-secret<your livekit api secret> --url=<your livekit url>
```

The environment variables `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and `LIVEKIT_URL` can be used instead of the cli-args.

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

<!--BEGIN_REPO_NAV-->
<br/><table>
<thead><tr><th colspan="2">LiveKit Ecosystem</th></tr></thead>
<tbody>
<tr><td>Real-time SDKs</td><td><a href="https://github.com/livekit/components-js">React Components</a> · <a href="https://github.com/livekit/client-sdk-js">JavaScript</a> · <a href="https://github.com/livekit/client-sdk-swift">iOS/macOS</a> · <a href="https://github.com/livekit/client-sdk-android">Android</a> · <a href="https://github.com/livekit/client-sdk-flutter">Flutter</a> · <a href="https://github.com/livekit/client-sdk-react-native">React Native</a> · <a href="https://github.com/livekit/client-sdk-rust">Rust</a> · <a href="https://github.com/livekit/client-sdk-python">Python</a> · <a href="https://github.com/livekit/client-sdk-unity-web">Unity (web)</a> · <a href="https://github.com/livekit/client-sdk-unity">Unity (beta)</a></td></tr><tr></tr>
<tr><td>Server APIs</td><td><a href="https://github.com/livekit/server-sdk-js">Node.js</a> · <a href="https://github.com/livekit/server-sdk-go">Golang</a> · <a href="https://github.com/livekit/server-sdk-ruby">Ruby</a> · <a href="https://github.com/livekit/server-sdk-kotlin">Java/Kotlin</a> · <a href="https://github.com/livekit/client-sdk-python">Python</a> · <a href="https://github.com/livekit/client-sdk-rust">Rust</a> · <a href="https://github.com/agence104/livekit-server-sdk-php">PHP (community)</a></td></tr><tr></tr>
<tr><td>Agents Frameworks</td><td><b>Python</b> · <a href="https://github.com/livekit/agent-playground">Playground</a></td></tr><tr></tr>
<tr><td>Services</td><td><a href="https://github.com/livekit/livekit">Livekit server</a> · <a href="https://github.com/livekit/egress">Egress</a> · <a href="https://github.com/livekit/ingress">Ingress</a> · <a href="https://github.com/livekit/sip">SIP</a></td></tr><tr></tr>
<tr><td>Resources</td><td><a href="https://docs.livekit.io">Docs</a> · <a href="https://github.com/livekit-examples">Example apps</a> · <a href="https://livekit.io/cloud">Cloud</a> · <a href="https://docs.livekit.io/oss/deployment">Self-hosting</a> · <a href="https://github.com/livekit/livekit-cli">CLI</a></td></tr>
</tbody>
</table>
<!--END_REPO_NAV-->
