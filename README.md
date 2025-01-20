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

## ✨ NEW ✨

### In-house phrase endpointing model

We’ve trained a new, open weights phrase endpointing model that significantly improves end-of-turn detection and conversational flow between voice agents and users by reducing agent interruptions. Optimized to run on CPUs, it’s available via [livekit-plugins-turn-detector](https://pypi.org/project/livekit-plugins-turn-detector/) package.

## What is Agents?

The **Agents framework** enables you to build AI-driven server programs that can see, hear, and speak in realtime. It offers a fully open-source platform for creating realtime, agentic applications.

## Features

- **Flexible integrations**: A comprehensive ecosystem to mix and match the right models for each use case.
- **AI voice agents**: `VoicePipelineAgent` and `MultimodalAgent` help orchestrate the conversation flow using LLMs and other AI models.
- **Integrated job scheduling**: Built-in task scheduling and distribution with [dispatch APIs](https://docs.livekit.io/agents/build/dispatch/) to connect end users to agents.
- **Realtime media transport**: Stream audio, video, and data over WebRTC and SIP with client SDKs for most platforms.
- **Telephony integration**: Works seamlessly with LiveKit's [telephony stack](https://docs.livekit.io/sip/), allowing your agent to make calls to or receive calls from phones.
- **Exchange data with clients**: Use [RPCs](https://docs.livekit.io/home/client/data/rpc/) and other [Data APIs](https://docs.livekit.io/home/client/data/) to seamlessly exchange data with clients.
- **Open-source**: Fully open-source, allowing you to run the entire stack on your own servers, including [LiveKit server](https://github.com/livekit/livekit), one of the most widely used WebRTC media servers.

<!--END_DESCRIPTION-->

## Installation

To install the core Agents library:

```bash
pip install livekit-agents
```

## Integrations

The framework includes a variety of plugins that make it easy to process streaming input or generate output. For example, there are plugins for converting text-to-speech or running inference with popular LLMs. Here's how you can install a plugin:

```bash
pip install livekit-plugins-openai
```

### Realtime API

We've partnered with OpenAI on a new `MultimodalAgent` API in the Agents framework. This class completely wraps OpenAI’s Realtime API, abstracts away the raw wire protocol, and provide an ultra-low latency WebRTC transport between GPT-4o and your users’ devices. This same stack powers Advanced Voice in the ChatGPT app.

- Try the Realtime API in our [playground](https://playground.livekit.io/) [[code](https://github.com/livekit-examples/realtime-playground)]
- Check out our [guide](https://docs.livekit.io/agents/openai) to building your first app with this new API

### LLM

| Provider        | Package                   | Usage                                                                                                                             |
| --------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| OpenAI          | livekit-plugins-openai    | [openai.LLM()](https://docs.livekit.io/python/livekit/plugins/openai/index.html#livekit.plugins.openai.LLM)                       |
| Azure OpenAI    | livekit-plugins-openai    | [openai.LLM.with_azure()](https://docs.livekit.io/python/livekit/plugins/openai/index.html#livekit.plugins.openai.LLM.with_azure) |
| Anthropic       | livekit-plugins-anthropic | [anthropic.LLM()](https://docs.livekit.io/python/livekit/plugins/anthropic/index.html#livekit.plugins.anthropic.LLM)              |
| Google (Gemini) | livekit-plugins-openai    | [openai.LLM.with_vertex()](https://docs.livekit.io/python/livekit/plugins/openai/#livekit.plugins.openai.LLM.with_vertex)         |
| Cerebras        | livekit-plugins-openai    | [openai.LLM.with_cerebras()](https://docs.livekit.io/python/livekit/plugins/openai/#livekit.plugins.openai.LLM.with_cerebras)     |
| Groq            | livekit-plugins-openai    | [openai.LLM.with_groq()](https://docs.livekit.io/python/livekit/plugins/openai/#livekit.plugins.openai.LLM.with_groq)             |
| Ollama          | livekit-plugins-openai    | [openai.LLM.with_ollama()](https://docs.livekit.io/python/livekit/plugins/openai/#livekit.plugins.openai.LLM.with_ollama)         |
| Perplexity      | livekit-plugins-openai    | [openai.LLM.with_perplexity()](https://docs.livekit.io/python/livekit/plugins/openai/#livekit.plugins.openai.LLM.with_perplexity) |
| Together.ai     | livekit-plugins-openai    | [openai.LLM.with_together()](https://docs.livekit.io/python/livekit/plugins/openai/#livekit.plugins.openai.LLM.with_together)     |
| X.ai (Grok)     | livekit-plugins-openai    | [openai.LLM.with_x_ai()](https://docs.livekit.io/python/livekit/plugins/openai/#livekit.plugins.openai.LLM.with_x_ai)             |

### STT

| Provider         | Package                    | Streaming | Usage                                                                                                                   |
| ---------------- | -------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------- |
| Azure            | livekit-plugins-azure      | ✅        | [azure.STT()](https://docs.livekit.io/python/livekit/plugins/azure/index.html#livekit.plugins.azure.STT)                |
| Deepgram         | livekit-plugins-deepgram   | ✅        | [deepgram.STT()](https://docs.livekit.io/python/livekit/plugins/deepgram/index.html#livekit.plugins.deepgram.STT)       |
| OpenAI (Whisper) | livekit-plugins-openai     |           | [openai.STT()](https://docs.livekit.io/python/livekit/plugins/openai/index.html#livekit.plugins.openai.STT)             |
| Google           | livekit-plugins-google     | ✅        | [google.STT()](https://docs.livekit.io/python/livekit/plugins/google/index.html#livekit.plugins.google.STT)             |
| AssemblyAI       | livekit-plugins-assemblyai |           | [assemblyai.STT()](https://docs.livekit.io/python/livekit/plugins/assemblyai/index.html#livekit.plugins.assemblyai.STT) |
| Groq (Whisper)   | livekit-plugins-openai     |           | [openai.STT.with_groq()](https://docs.livekit.io/python/livekit/plugins/openai/#livekit.plugins.openai.STT.with_groq)   |
| FAL (Whizper)    | livekit-plugins-fal        |           | [fal.STT()](https://docs.livekit.io/python/livekit/plugins/fal/index.html#livekit.plugins.fal.STT)                      |

### TTS

| Provider     | Package                    | Streaming | Voice Cloning | Usage                                                                                                                   |
| ------------ | -------------------------- | --------- | ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Cartesia     | livekit-plugins-cartesia   | ✅        | ✅            | [cartesia.TTS()](https://docs.livekit.io/python/livekit/plugins/cartesia/index.html#livekit.plugins.cartesia.TTS)       |
| ElevenLabs   | livekit-plugins-elevenlabs | ✅        | ✅            | [elevenlabs.TTS()](https://docs.livekit.io/python/livekit/plugins/elevenlabs/index.html#livekit.plugins.elevenlabs.TTS) |
| OpenAI       | livekit-plugins-openai     |           |               | [openai.TTS()](https://docs.livekit.io/python/livekit/plugins/openai/index.html#livekit.plugins.openai.TTS)             |
| Azure OpenAI | livekit-plugins-openai     |           |               | [openai.TTS.with_azure()](https://docs.livekit.io/python/livekit/plugins/openai/#livekit.plugins.openai.TTS.with_azure) |
| Google       | livekit-plugins-google     | ✅        | ✅            | [google.TTS()](https://docs.livekit.io/python/livekit/plugins/google/index.html#livekit.plugins.google.TTS)             |
| Deepgram     | livekit-plugins-deepgram   | ✅        |               | [deepgram.TTS()](https://docs.livekit.io/python/livekit/plugins/deepgram/index.html#livekit.plugins.deepgram.TTS)       |

### Other plugins

| Plugin                        | Description                         |
| ----------------------------- | ----------------------------------- |
| livekit-plugins-rag           | Annoy based simple RAG              |
| livekit-plugins-llama-index   | RAG with LlamaIndex                 |
| livekit-plugins-nltk          | Utilities for working with text     |
| livekit-plugins-vad           | Voice activity detection            |
| livekit-plugins-turn-detector | Conversational turn detection model |

## Documentation and guides

Documentation on the framework and how to use it can be found [here](https://docs.livekit.io/agents)

## Example agents

| Description                                                           | Demo Link                                      | Code Link                                                                                                     |
| --------------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| A basic voice agent using a pipeline of STT, LLM, and TTS             | [demo](https://kitt.livekit.io)                | [code](https://github.com/livekit/agents/blob/main/examples/voice-pipeline-agent/minimal_assistant.py)        |
| Voice agent using the new OpenAI Realtime API                         | [demo](https://playground.livekit.io)          | [code](https://github.com/livekit-examples/realtime-playground)                                               |
| Super fast voice agent using Cerebras hosted Llama 3.1                | [demo](https://cerebras.vercel.app)            | [code](https://github.com/dsa/fast-voice-assistant/)                                                          |
| Voice agent using Cartesia's Sonic model                              | [demo](https://cartesia-assistant.vercel.app/) | [code](https://github.com/livekit-examples/cartesia-voice-agent)                                              |
| Agent that looks up the current weather via function call             | N/A                                            | [code](https://github.com/livekit/agents/blob/main/examples/voice-pipeline-agent/function_calling_weather.py) |
| Voice Agent using Gemini 2.0 Flash                                    | N/A                                            | [code](https://github.com/livekit-examples/voice-pipeline-agent/gemini_voice_agent.py)                        |
| Voice agent with custom turn-detection model                          | N/A                                            | [code](https://github.com/livekit/agents/blob/main/examples/voice-pipeline-agent/turn_detector.py)            |
| Voice agent that performs a RAG-based lookup                          | N/A                                            | [code](https://github.com/livekit/agents/tree/main/examples/voice-pipeline-agent/simple-rag)                  |
| Simple agent that echos back the last utterance                       | N/A                                            | [code](https://github.com/livekit/agents/tree/main/examples/echo-agent)                                       |
| Video agent that publishes a stream of RGB frames                     | N/A                                            | [code](https://github.com/livekit/agents/tree/main/examples/simple-color)                                     |
| Transcription agent that generates text captions from a user's speech | N/A                                            | [code](https://github.com/livekit/agents/tree/main/examples/speech-to-text)                                   |
| A chat agent you can text who will respond back with generated speech | N/A                                            | [code](https://github.com/livekit/agents/tree/main/examples/text-to-speech)                                   |
| Localhost multi-agent conference call                                 | N/A                                            | [code](https://github.com/dsa/multi-agent-meeting)                                                            |
| Moderation agent that uses Hive to detect spam/abusive video          | N/A                                            | [code](https://github.com/dsa/livekit-agents/tree/main/hive-moderation-agent)                                 |

## Contributing

The Agents framework is under active development in a rapidly evolving field. We welcome and appreciate contributions of any kind, be it feedback, bugfixes, features, new plugins and tools, or better documentation. You can file issues under this repo, open a PR, or chat with us in LiveKit's [Slack community](https://livekit.io/join-slack).

<!--BEGIN_REPO_NAV-->

<br/><table>

<thead><tr><th colspan="2">LiveKit Ecosystem</th></tr></thead>
<tbody>
<tr><td>Realtime SDKs</td><td><a href="https://github.com/livekit/client-sdk-js">Browser</a> · <a href="https://github.com/livekit/client-sdk-swift">iOS/macOS/visionOS</a> · <a href="https://github.com/livekit/client-sdk-android">Android</a> · <a href="https://github.com/livekit/client-sdk-flutter">Flutter</a> · <a href="https://github.com/livekit/client-sdk-react-native">React Native</a> · <a href="https://github.com/livekit/rust-sdks">Rust</a> · <a href="https://github.com/livekit/node-sdks">Node.js</a> · <a href="https://github.com/livekit/python-sdks">Python</a> · <a href="https://github.com/livekit/client-sdk-unity">Unity</a> · <a href="https://github.com/livekit/client-sdk-unity-web">Unity (WebGL)</a></td></tr><tr></tr>
<tr><td>Server APIs</td><td><a href="https://github.com/livekit/node-sdks">Node.js</a> · <a href="https://github.com/livekit/server-sdk-go">Golang</a> · <a href="https://github.com/livekit/server-sdk-ruby">Ruby</a> · <a href="https://github.com/livekit/server-sdk-kotlin">Java/Kotlin</a> · <a href="https://github.com/livekit/python-sdks">Python</a> · <a href="https://github.com/livekit/rust-sdks">Rust</a> · <a href="https://github.com/agence104/livekit-server-sdk-php">PHP (community)</a></td></tr><tr></tr>
<tr><td>UI Components</td><td><a href="https://github.com/livekit/components-js">React</a> · <a href="https://github.com/livekit/components-android">Android Compose</a> · <a href="https://github.com/livekit/components-swift">SwiftUI</a></td></tr><tr></tr>
<tr><td>Agents Frameworks</td><td><b>Python</b> · <a href="https://github.com/livekit/agents-js">Node.js</a> · <a href="https://github.com/livekit/agent-playground">Playground</a></td></tr><tr></tr>
<tr><td>Services</td><td><a href="https://github.com/livekit/livekit">LiveKit server</a> · <a href="https://github.com/livekit/egress">Egress</a> · <a href="https://github.com/livekit/ingress">Ingress</a> · <a href="https://github.com/livekit/sip">SIP</a></td></tr><tr></tr>
<tr><td>Resources</td><td><a href="https://docs.livekit.io">Docs</a> · <a href="https://github.com/livekit-examples">Example apps</a> · <a href="https://livekit.io/cloud">Cloud</a> · <a href="https://docs.livekit.io/home/self-hosting/deployment">Self-hosting</a> · <a href="https://github.com/livekit/livekit-cli">CLI</a></td></tr>
</tbody>
</table>
<!--END_REPO_NAV-->
