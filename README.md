<!--BEGIN_BANNER_IMAGE-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/.github/banner_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="/.github/banner_light.png">
  <img style="width:100%;" alt="The LiveKit icon, the name of the repository and some sample code in the background." src="https://raw.githubusercontent.com/livekit/agents/main/.github/banner_light.png">
</picture>

<!--END_BANNER_IMAGE-->

<!--BEGIN_DESCRIPTION-->
<br /><br />
Looking for the JS/TS library? Check out [AgentsJS](https://github.com/livekit/agents-js)

The Agents framework allows you to build AI-driven server programs that can see, hear, and speak in realtime. Your agent connects with end user devices through a LiveKit session. During that session, your agent can process text, audio, images, or video streaming from a user's device, and have an AI model generate any combination of those same modalities as output, and stream them back to the user. 

## Features
- Plugins for popular LLMs, transcription and text-to-speech services, and RAG databases
- High-level abstractions for building voice agents or assistants with automatic turn detection, interruption handling, function calling, and transcriptions
- Compatible with LiveKit's [telephony stack](https://github.com/livekit/sip), allowing your agent to make calls to or receive calls from phones
- Integrated load balancing system that manages pools of agents with edge-based dispatch, monitoring, and transparent failover
- Running your agents is identical across localhost, [self-hosted](https://github.com/livekit/livekit), and [LiveKit Cloud](https://cloud.livekit.io) environments

<!--END_DESCRIPTION-->

## Installation

To install the core Agents library:

```bash
pip install livekit-agents
```

## Plugins

The framework includes a variety of plugins that make it easy to process streaming input or generate output. For example, there are plugins for converting text-to-speech or running inference with popular LLMs. Here's how you can install a plugin:

```bash
pip install livekit-plugins-openai
```

The following plugins are available today:

| Plugin                                                                             | Features                                    |
| ---------------------------------------------------------------------------------- | ------------------------------------------- |
| [livekit-plugins-anthropic](https://pypi.org/project/livekit-plugins-anthropic/)   | LLM                                         |
| [livekit-plugins-azure](https://pypi.org/project/livekit-plugins-azure/)           | STT, TTS                                    |
| [livekit-plugins-deepgram](https://pypi.org/project/livekit-plugins-deepgram/)     | STT                                         |
| [livekit-plugins-cartesia](https://pypi.org/project/livekit-plugins-cartesia/)     | TTS                                         |
| [livekit-plugins-elevenlabs](https://pypi.org/project/livekit-plugins-elevenlabs/) | TTS                                         |
| [livekit-plugins-playht](https://pypi.org/project/livekit-plugins-playht/)         | TTS                                         |
| [livekit-plugins-google](https://pypi.org/project/livekit-plugins-google/)         | STT, TTS                                    |
| [livekit-plugins-nltk](https://pypi.org/project/livekit-plugins-nltk/)             | Utilities for working with text             |
| [livekit-plugins-rag](https://pypi.org/project/livekit-plugins-rag/)               | Utilities for performing RAG                |
| [livekit-plugins-openai](https://pypi.org/project/livekit-plugins-openai/)         | LLM, STT, TTS, Assistants API, Realtime API |
| [livekit-plugins-silero](https://pypi.org/project/livekit-plugins-silero/)         | VAD                                         |


## Documentation and guides

Documentation on the framework and how to use it can be found [here](https://docs.livekit.io/agents)

## Example agents

- A basic voice agent using a pipeline of STT, LLM, and TTS [[Demo](https://kitt.livekit.io) | [Code](https://github.com/livekit/agents/blob/main/examples/voice-assistant/minimal_assistant.py)]]
- Super fast voice agent using Cerebras hosted Llama 3.1 [[Demo](https://cerebras.vercel.app) | [Code](https://github.com/dsa/fast-voice-assistant/)]
- Voice agent using Cartesia's Sonic model [[Demo](https://cartesia-assistant.vercel.app/)]
- Agent that looks up the current weather via function call [[Code](https://github.com/livekit/agents/blob/main/examples/voice-assistant/function_calling_weather.py)]
- Voice agent that performs a RAG-based lookup [[Code](https://github.com/livekit/agents/tree/main/examples/voice-assistant/simple-rag)]
- Video agent that publishes a stream of RGB frames [[Code](https://github.com/livekit/agents/tree/main/examples/simple-color)]
- Transcription agent that generates text captions from a user's speech [[Code](https://github.com/livekit/agents/tree/main/examples/speech-to-text)]
- A chat agent you can text who will respond back with genereated speech [[Code](https://github.com/livekit/agents/tree/main/examples/text-to-speech)]
- Localhost multi-agent conference call [[Code](https://github.com/dsa/multi-agent-meeting)]
- Moderation agent that uses Hive to detect spam/abusive video [[Code](https://github.com/dsa/livekit-agents/tree/main/hive-moderation-agent)]

## Contributing

The Agents framework is under active development in a rapidly evolving field. We welcome and appreciate contributions of any kind, be it feedback, bugfixes, features, new plugins and tools, or better documentation. You can file issues under this repo, open a PR, or chat with us in LiveKit's [Slack community](https://livekit.io/join-slack).

<!--BEGIN_REPO_NAV-->

<br/><table>

<thead><tr><th colspan="2">LiveKit Ecosystem</th></tr></thead>
<tbody>
<tr><td>Realtime SDKs</td><td><a href="https://github.com/livekit/components-js">React Components</a> · <a href="https://github.com/livekit/client-sdk-js">Browser</a> · <a href="https://github.com/livekit/components-swift">Swift Components</a> · <a href="https://github.com/livekit/client-sdk-swift">iOS/macOS/visionOS</a> · <a href="https://github.com/livekit/client-sdk-android">Android</a> · <a href="https://github.com/livekit/client-sdk-flutter">Flutter</a> · <a href="https://github.com/livekit/client-sdk-react-native">React Native</a> · <a href="https://github.com/livekit/rust-sdks">Rust</a> · <a href="https://github.com/livekit/node-sdks">Node.js</a> · <a href="https://github.com/livekit/python-sdks">Python</a> · <a href="https://github.com/livekit/client-sdk-unity-web">Unity (web)</a> · <a href="https://github.com/livekit/client-sdk-unity">Unity (beta)</a></td></tr><tr></tr>
<tr><td>Server APIs</td><td><a href="https://github.com/livekit/node-sdks">Node.js</a> · <a href="https://github.com/livekit/server-sdk-go">Golang</a> · <a href="https://github.com/livekit/server-sdk-ruby">Ruby</a> · <a href="https://github.com/livekit/server-sdk-kotlin">Java/Kotlin</a> · <a href="https://github.com/livekit/python-sdks">Python</a> · <a href="https://github.com/livekit/rust-sdks">Rust</a> · <a href="https://github.com/agence104/livekit-server-sdk-php">PHP (community)</a></td></tr><tr></tr>
<tr><td>Agents Frameworks</td><td><b>Python</b> · <a href="https://github.com/livekit/agent-playground">Playground</a></td></tr><tr></tr>
<tr><td>Services</td><td><a href="https://github.com/livekit/livekit">LiveKit server</a> · <a href="https://github.com/livekit/egress">Egress</a> · <a href="https://github.com/livekit/ingress">Ingress</a> · <a href="https://github.com/livekit/sip">SIP</a></td></tr><tr></tr>
<tr><td>Resources</td><td><a href="https://docs.livekit.io">Docs</a> · <a href="https://github.com/livekit-examples">Example apps</a> · <a href="https://livekit.io/cloud">Cloud</a> · <a href="https://docs.livekit.io/home/self-hosting/deployment">Self-hosting</a> · <a href="https://github.com/livekit/livekit-cli">CLI</a></td></tr>
</tbody>
</table>
<!--END_REPO_NAV-->
