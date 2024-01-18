<!--BEGIN_BANNER_IMAGE-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/.github/banner_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="/.github/banner_light.png">
  <img style="width:100%;" alt="The LiveKit icon, the name of the repository and some sample code in the background." src="https://raw.githubusercontent.com/livekit/agents/main/.github/banner_light.png">
</picture>

<!--END_BANNER_IMAGE-->

# LiveKit Agents

<!--BEGIN_DESCRIPTION-->

The Agent Framework is designed for building real-time, programmable participants
that run on servers. Easily tap into LiveKit WebRTC sessions and process or generate
audio, video, and data streams.

<!--END_DESCRIPTION-->

The framework includes plugins for common workflows, such as voice activity detection and speech-to-text.

Agents integrates seamlessly with [LiveKit server](https://github.com/livekit/livekit), offloading job queuing and scheduling responsibilities to it. This eliminates the need for additional queuing infrastructure. Agent code developed on your local machine can scale to support thousands of concurrent sessions when deployed to a server in production.

> This SDK is currently in Developer Preview mode and not ready for production use. There will be bugs and APIs may change during this period.
>
> We welcome and appreciate any feedback or contributions. You can create issues here or chat live with us in the #dev channel within the [LiveKit Community Slack](https://livekit.io/join-slack).

## Docs & Guides

- [Overview](https://docs.livekit.io/agents/)
- [Quickstart](https://docs.livekit.io/agents/quickstart)
- [Working with plugins](https://docs.livekit.io/agents/plugins)
- [Deploying agents](https://docs.livekit.io/agents/deployment)

## Examples

### KITT

An voice assistant using DeepGram STT, GPT-4, and ElevenLabs TTS

- [Demo](https://kitt.livekit.io)
- [Source Code](https://github.com/livekit/agents/tree/main/examples/kitt)

## Installation

To install the core Agents library:

```bash
pip install livekit-agents
```

Agents includes a set of prebuilt plugins that make it easier to compose together agents. These plugins cover common tasks like converting speech to text or vice versa and running inference on a generative AI model. The following plugins are available today:

| Plugin                                                                             | Features                        |
| ---------------------------------------------------------------------------------- | ------------------------------- |
| [livekit-plugins-deepgram](https://pypi.org/project/livekit-plugins-deepgram/)     | STT                             |
| [livekit-plugins-directai](https://pypi.org/project/livekit-plugins-directai/)     | Vision, object detection        |
| [livekit-plugins-elevenlabs](https://pypi.org/project/livekit-plugins-elevenlabs/) | TTS                             |
| [livekit-plugins-fal](https://pypi.org/project/livekit-plugins-fal/)               | Image generation                |
| [livekit-plugins-google](https://pypi.org/project/livekit-plugins-google/)         | STT                             |
| [livekit-plugins-nltk](https://pypi.org/project/livekit-plugins-nltk/)             | Utilities for working with text |
| [livekit-plugins-openai](https://pypi.org/project/livekit-plugins-openai/)         | Dalle 3, STT, TTS               |
| [livekit-plugins-silero](https://pypi.org/project/livekit-plugins-silero/)         | VAD                             |

## Concepts

- **Agent**: A function that defines the workflow of a programmable, server-side participant. This is your application code.
- **Worker**: A container process responsible for managing job queuing with LiveKit server. Each worker is capable of running multiple agents simultaneously.
- **Plugin**: A library class that performs a specific task, like speech-to-text, from a specific provider. An agent can compose multiple plugins together to perform more complex tasks.

## Running an agent

The framework exposes a CLI interface to run your agent. To get started, you'll need the following environment variables set:

- LIVEKIT_URL
- LIVEKIT_API_KEY
- LIVEKIT_API_SECRET

### Running the worker

This will start the worker and wait for users to connect to your LiveKit server:

```bash
python my_agent.py start
```

### Joining a specific room

To join a LiveKit room that's already active, you can use the `simulate-job` command:

```bash
python my_agent.py simulate-job --room-name <my-room>
```

### What happens when I run my agent?

When you follow the steps above to run your agent, a worker is started that opens an authenticated WebSocket connection to a LiveKit server instance(defined by your `LIVEKIT_URL` and authenticated with an access token).

No agents are actually running at this point. Instead, the worker is waiting for LiveKit server to give it a job.

When a room is created, the server notifies one of the registered workers about a new job. The notified worker can decide whether or not to accept it. If the worker accepts the job, the worker will instantiate your agent as a participant and have it join the room where it can start subscribing to tracks. A worker can manage multiple agent instances simultaneously.

If a notified worker rejects the job or does not accept within a predetermined timeout period, the server will route the job request to another available worker.

### What happens when I SIGTERM a worker?

The orchestration system was designed for production use cases. Unlike the typical web server, an agent is a stateful program, so it's important that a worker can't be terminated while it's managing any active agents.

When calling SIGTERM on a worker, the worker will signal to LiveKit server that it no longer wants additional jobs. It will also auto-reject any new job requests that get through before the server signal is received. The worker will remain alive while it manages any agents connected to rooms.

### Downloading model files

Some plugins require model files to be downloaded before they can be used. To download all the necessary models for your agent, execute the following command:

```bash
python my_agent.py download-files
```

If you're developing a custom plugin, you can integrate this functionality by implementing a `download_files` method in your Plugin class:

```python
class MyPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__)

    def download_files(self):
        _ = torch.hub.load(
            repo_or_dir="my-repo",
            model="my-model",
        )
```

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
