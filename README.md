<!--BEGIN_BANNER_IMAGE-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/.github/banner_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="/.github/banner_light.png">
  <img style="width:100%;" alt="The LiveKit icon, the name of the repository and some sample code in the background." src="https://raw.githubusercontent.com/livekit/agents/main/.github/banner_light.png">
</picture>

<!--END_BANNER_IMAGE-->

# LiveKit Agents

<!--BEGIN_DESCRIPTION-->

LiveKit Agents is an end-to-end framework for building real-time, multimodal AI "agents" that interact with end-users through voice, video, and data channels. This framework allows you to build an agent using Python.

<!--END_DESCRIPTION-->

The framework includes plugins for common workflows, such as voice activity detection and speech-to-text.

Furthermore, it integrates seamlessly with LiveKit server, offloading job queuing and scheduling responsibilities to it. This approach eliminates the need for additional queuing infrastructure. The code developed on your local machine is fully scalable when deployed to a server, supporting thousands of concurrent sessions.

## Docs & Guides

- [Overview](https://docs.livekit.io/agents/)
- [Quickstart](https://docs.livekit.io/agents/quickstart)
- [Working with plugins](https://docs.livekit.io/agents/plugins)
- [Deploying agents](https://docs.livekit.io/agents/deployment)

## Examples

### KITT

An voice assistant using DeepGram STT, ChatGPT-4, and ElevenLabs TTS

- [Live Demo](https://kitt.livekit.io)
- [Source Code](https://github.com/livekit/agents/tree/main/examples/kitt)

## Installation

To install the core agent library:

```bash
pip install livekit-agents
```

Agent Framework includes a set of prebuilt plugins that make it easier to compose together agents. These plugins cover common workflows involving speech-to-text, text-to-speech, and image generation. The following plugins are available today:

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

- **Agent**: A function that defines the workflow of the server-side participant. This is what you will be developing.
- **Worker**: A container process responsible for managing job queuing with LiveKit server. Each worker is capable of running multiple agents simultaneously.
- **Plugin**: A library class that perform a specific task like speech-to-text with a specific provider. Agents can compose multiple plugins together to perform more complex tasks.

## Running an Agent

The Agent Framework expose a CLI interface to run your agent. To get started, you'll need the following environment variables set:

- LIVEKIT_URL
- LIVEKIT_API_KEY
- LIVEKIT_API_SECRET

### Running in worker mode

This will start the worker and wait for users to connect to your LiveKit server:

```bash
python my_agent.py start
```

### Joining a specific room

To join a particular LiveKit room that's already active, you can use the `simulate-job` command:

```bash
python my_agent.py simulate-job --room-name <my-room>
```

### What is happening when I run my Agent?

When you run your agent with the above commands, a worker is started that opens an authenticated websocket connection to a LiveKit server (defined by your `LIVEKIT_URL` and authenticated with an access token).

This doesn't actually run any agents. Instead, the worker sits waiting for LiveKit server to give it a job.

When a room is created, LiveKit server notifies one of the registered workers about the job. The first worker to accept a job will instantiate your agent and have it join the room. A worker can manage multiple agent instances simultaneously.

Once the LiveKit server is given a job, the worker can decide whether or not to accept it. Accepting the job will create a LiveKit participant that joins the room and begin subscribing to tracks.

### What happens when I SIGTERM one of my Workers?

The framework was designed for production use cases. Since agents are more stateful entities than typical web-servers, it's important that workers can't be terminated while they are running active agents.

When calling SIGTERM on a worker, the worker will signal to the LiveKit server that it does not want to be given any more jobs. It will also auto-decline any new job requests that might sneak in before the server signaling has occurred. The worker will remain alive while it manages an agent that is still connected to a room.

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
<tr><td>Agents</td><td><b>Python</b> · <a href="https://github.com/livekit/agent-playground">Playground</a></td></tr><tr></tr>
<tr><td>Services</td><td><a href="https://github.com/livekit/livekit">Livekit server</a> · <a href="https://github.com/livekit/egress">Egress</a> · <a href="https://github.com/livekit/ingress">Ingress</a> · <a href="https://github.com/livekit/sip">SIP</a></td></tr><tr></tr>
<tr><td>Resources</td><td><a href="https://docs.livekit.io">Docs</a> · <a href="https://github.com/livekit-examples">Example apps</a> · <a href="https://livekit.io/cloud">Cloud</a> · <a href="https://docs.livekit.io/oss/deployment">Self-hosting</a> · <a href="https://github.com/livekit/livekit-cli">CLI</a></td></tr>
</tbody>
</table>
<!--END_REPO_NAV-->
