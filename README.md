# SalesCode.ai Final Round Qualifier Submission
**Candidate:** Kritnandan
**Branch:** `feature/livekit-interrupt-handler-Kritnandan`

This document details the implementation of the LiveKit Voice Interruption Handling Challenge.

---

## What Changed

The core logic was integrated into the existing agent event loop by modifying three Python files and adding one necessary model file.

* **`livekit-agents/livekit/agents/agent.py`**
    * The `Agent` class constructor (`__init__`) was modified to accept a new `ignored_fillers: NotGivenOr[List[str] | None]` argument.
    * A public `@property` named `ignored_fillers` was added to expose this list to the agent's runtime.
    * (Bonus) A public method `async def update_ignored_fillers(self, new_fillers: list[str])` was added to support dynamic runtime updates.

* **`livekit-agents/livekit/agents/agent_activity.py`**
    * This file contains the core "extension layer" logic.
    * `__init__`: Now reads the `ignored_fillers` list from the `Agent` object and stores it in a `set` (`self._ignored_fillers`) for efficient O(1) lookups.
    * `on_start_of_speech`: Modified to set a `self._pending_interruption = True` flag, rather than immediately interrupting.
    * `on_vad_inference_done`: This function's call to `_interrupt_by_audio_activity()` was removed to **disable** the default VAD-only interruption, ensuring only STT-verified speech causes a stop.
    * `_is_filler_only(self, text: str) -> bool`: A new helper method was added. It uses `startswith()` logic to robustly check if a transcribed word (e.g., "ummm") matches a root filler (e.g., "um").
    * `on_interim_transcript`: This is the new decision-making hub. It checks for three scenarios:
        1.  **Filler:** If `_is_filler_only` is `True`, the speech is logged and ignored.
        2.  **Murmur:** If ASR `confidence < 0.5`, the speech is logged as low-confidence and ignored.
        3.  **Real Interruption:** If it's a non-filler with high confidence, `_interrupt_by_audio_activity()` is called.
    * `on_final_transcript`: Modified to also use the `_is_filler_only` check to handle cases where only a final (non-interim) transcript is received.

* **`examples/voice_agents/basic_agent.py`**
    * Now reads `IGNORED_FILLERS` from a `.env` file using `os.environ.get()` and `json.loads()`.
    * The `MyAgent` class constructor is updated to pass this list to the base `Agent`.
    * (Bonus) A `@function_tool` named `update_filler_words` was added to demonstrate the dynamic runtime update bonus challenge.

* **`livekit-plugins/livekit-plugins-silero/livekit/plugins/silero/resources/silero_vad.onnx`**
    * This VAD model file was added to the repository. It was found to be missing and is required to "ensure the agent runs end-to-end" per the submission requirements.

---

## What Works

* **Filler Ignoring:** The agent correctly ignores filler words (e.g., "uh", "umm", "haan") while it is speaking, as verified by live testing.
* **Real Interruption:** The agent stops *immediately* when a non-filler word (e.g., "wait", "stop") is detected with high confidence.
* **Murmur Ignoring:** Background noise and coughs transcribed with low confidence (e.g., "yeah" at 0.3) are successfully ignored.
* **Normal Conversation:** When the agent is quiet, it correctly registers "umm" (or any other filler) as the start of a user turn.
* **Configurability:** The list of fillers is fully configurable via the `.env` file, as demonstrated by the `basic_agent.py` implementation.
* **Multi-Language (Bonus):** The system is language-agnostic. By adding Hindi words like `"haan"` and `"accha"` to the `.env` file, it correctly ignores them.
* **Dynamic Updates (Bonus):** The new `update_filler_words` tool allows the filler list to be changed live during a conversation.

---

## Known Issues

* The system's robustness is highly dependent on the ASR's transcription quality and its confidence scoring. If the ASR (e.g., AssemblyAI) provides an incorrect high-confidence score for a murmur, it may cause a false interruption.
* There is a slight, unavoidable delay in "real-time" interruptions, as the system must wait for the ASR to process the audio and return the first interim transcript.

---

## Steps to Test

1.  **Environment Setup:**
    * Ensure Python 3.11+ is installed.
    * Create and activate a virtual environment:
        ```bash
        python3.12 -m venv .venv
        source .venv/bin/activate
        ```

2.  **Configure Environment:**
    * Create a file named `.env` in the root `agents` directory.
    * Add your API keys (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `ASSEMBLYAI_API_KEY`, `OPENAI_API_KEY`, `CARTESIA_API_KEY`).
    * Add the multi-language filler list to your `.env` file:
        ```ini
        IGNORED_FILLERS='["uh", "um", "hm", "haan", "accha", "like", "you know"]'
        ```

3.  **Install Dependencies:**
    * Install the modified local `livekit-agents` library:
        ```bash
        pip install -e livekit-agents
        ```
    * Install the local plugins:
        ```bash
        pip install -e livekit-plugins/livekit-plugins-silero
        pip install -e livekit-plugins/livekit-plugins-turn-detector
        ```
    * Install the other required plugins:
        ```bash
        pip install livekit-plugins-openai livekit-plugins-cartesia livekit-plugins-assemblyai python-dotenv
        ```

4.  **Run the Agent:**
    * Run the agent with the `start` command:
        ```bash
        python examples/voice_agents/basic_agent.py start
        ```

5.  **Join the Room:**
    * Go to your LiveKit Cloud dashboard, click on **"Sandbox"**, and join a **"Video conference"** room. The agent (named `agent-AJ_...`) will join automatically.

6.  **Test Scenarios:**
    * **Test 1 (Filler Ignore):** While the agent speaks, say "ummm" or "haan".
        * **Expected:** Agent ignores you and continues speaking. Your terminal logs will show `DEBUG - Ignored filler interruption...`
    * **Test 2 (Real Interruption):** While the agent speaks, say "wait a second" or "stop".
        * **Expected:** Agent stops immediately. Your logs will show `INFO - Valid interruption detected...`
    * **Test 3 (Murmur Ignore):** Mumble, cough, or make a short noise.
        * **Expected:** Agent ignores you. Logs will show `DEBUG - Ignored low-confidence murmur...`
    * **Test 4 (Normal Filler):** While the agent is quiet, say "umm...".
        * **Expected:** Agent treats this as a normal turn and responds.

---

## Environment Details

* **Python Version:** 3.12 (Tested, 3.11+ recommended)
* **Dependencies:** `livekit-agents` (from local branch), `livekit-plugins-silero`, `livekit-plugins-turn-detector`, `livekit-plugins-openai`, `livekit-plugins-cartesia`, `livekit-plugins-assemblyai`, `python-dotenv`
* **Config:** All API keys and the `IGNORED_FILLERS` list are configured via a `.env` file in the project root.

---

*(Original LiveKit Agents README.md content follows below...)*

<!--BEGIN_BANNER_IMAGE-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/.github/banner_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="/.github/banner_light.png">
  <img style="width:100%;" alt="The LiveKit icon, the name of the repository and some sample code in the background." src="https://raw.githubusercontent.com/livekit/agents/main/.github/banner_light.png">
</picture>

<!--END_BANNER_IMAGE-->
<br />

![PyPI - Version](https://img.shields.io/pypi/v/livekit-agents)
[![PyPI Downloads](https://static.pepy.tech/badge/livekit-agents/month)](https://pepy.tech/projects/livekit-agents)
[![Slack community](https://img.shields.io/endpoint?url=https%3A%2F%2Flivekit.io%2Fbadges%2Fslack)](https://livekit.io/join-slack)
[![Twitter Follow](https://img.shields.io/twitter/follow/livekit)](https://twitter.com/livekit)
[![Ask DeepWiki for understanding the codebase](https://deepwiki.com/badge.svg)](https://deepwiki.com/livekit/agents)
[![License](https://img.shields.io/github/license/livekit/livekit)](https://github.com/livekit/livekit/blob/master/LICENSE)

<br />

Looking for the JS/TS library? Check out [AgentsJS](https://github.com/livekit/agents-js)

## What is Agents?

<!--BEGIN_DESCRIPTION-->

The Agent Framework is designed for building realtime, programmable participants
that run on servers. Use it to create conversational, multi-modal voice
agents that can see, hear, and understand.

<!--END_DESCRIPTION-->

## Features

- **Flexible integrations**: A comprehensive ecosystem to mix and match the right STT, LLM, TTS, and Realtime API to suit your use case.
- **Integrated job scheduling**: Built-in task scheduling and distribution with [dispatch APIs](https://docs.livekit.io/agents/build/dispatch/) to connect end users to agents.
- **Extensive WebRTC clients**: Build client applications using LiveKit's open-source SDK ecosystem, supporting all major platforms.
- **Telephony integration**: Works seamlessly with LiveKit's [telephony stack](https://docs.livekit.io/sip/), allowing your agent to make calls to or receive calls from phones.
- **Exchange data with clients**: Use [RPCs](https://docs.livekit.io/home/client/data/rpc/) and other [Data APIs](https://docs.livekit.io/home/client/data/) to seamlessly exchange data with clients.
- **Semantic turn detection**: Uses a transformer model to detect when a user is done with their turn, helps to reduce interruptions.
- **MCP support**: Native support for MCP. Integrate tools provided by MCP servers with one loc.
- **Builtin test framework**: Write tests and use judges to ensure your agent is performing as expected.
- **Open-source**: Fully open-source, allowing you to run the entire stack on your own servers, including [LiveKit server](https://github.com/livekit/livekit), one of the most widely used WebRTC media servers.

## Installation

To install the core Agents library, along with plugins for popular model providers:

```bash
pip install "livekit-agents[openai,silero,deepgram,cartesia,turn-detector]~=1.0"
```

## Docs and guides

Documentation on the framework and how to use it can be found [here](https://docs.livekit.io/agents/)

## Core concepts

- Agent: An LLM-based application with defined instructions.
- AgentSession: A container for agents that manages interactions with end users.
- entrypoint: The starting point for an interactive session, similar to a request handler in a web server.
- Worker: The main process that coordinates job scheduling and launches agents for user sessions.

## Usage

### Simple voice agent

---

```python
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import deepgram, elevenlabs, openai, silero

@function_tool
async def lookup_weather(
    context: RunContext,
    location: str,
):
    """Used to look up weather information."""

    return {"weather": "sunny", "temperature": 70}


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = Agent(
        instructions="You are a friendly voice assistant built by LiveKit.",
        tools=[lookup_weather],
    )
    session = AgentSession(
        vad=silero.VAD.load(),
        # any combination of STT, LLM, TTS, or realtime API can be used
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(),
    )

    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="greet the user and ask about their day")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

You'll need the following environment variables for this example:

- DEEPGRAM_API_KEY
- OPENAI_API_KEY
- ELEVEN_API_KEY

### Multi-agent handoff

---

This code snippet is abbreviated. For the full example, see [multi_agent.py](examples/voice_agents/multi_agent.py)

```python
...
class IntroAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"You are a story teller. Your goal is to gather a few pieces of information from the user to make the story personalized and engaging."
            "Ask the user for their name and where they are from"
        )

    async def on_enter(self):
        self.session.generate_reply(instructions="greet the user and gather information")

    @function_tool
    async def information_gathered(
        self,
        context: RunContext,
        name: str,
        location: str,
    ):
        """Called when the user has provided the information needed to make the story personalized and engaging.

        Args:
            name: The name of the user
            location: The location of the user
        """

        context.userdata.name = name
        context.userdata.location = location

        story_agent = StoryAgent(name, location)
        return story_agent, "Let's start the story!"


class StoryAgent(Agent):
    def __init__(self, name: str, location: str) -> None:
        super().__init__(
            instructions=f"You are a storyteller. Use the user's information in order to make the story personalized."
            f"The user's name is {name}, from {location}"
            # override the default model, switching to Realtime API from standard LLMs
            llm=openai.realtime.RealtimeModel(voice="echo"),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    userdata = StoryData()
    session = AgentSession[StoryData](
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="echo"),
        userdata=userdata,
    )

    await session.start(
        agent=IntroAgent(),
        room=ctx.room,
    )
...
```

### Testing

Automated tests are essential for building reliable agents, especially with the non-deterministic behavior of LLMs. LiveKit Agents include native test integration to help you create dependable agents.

```python
@pytest.mark.asyncio
async def test_no_availability() -> None:
    llm = google.LLM()
    async AgentSession(llm=llm) as sess:
        await sess.start(MyAgent())
        result = await sess.run(
            user_input="Hello, I need to place an order."
        )
        result.expect.skip_next_event_if(type="message", role="assistant")
        result.expect.next_event().is_function_call(name="start_order")
        result.expect.next_event().is_function_call_output()
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(llm, intent="assistant should be asking the user what they would like")
        )

```

## Examples

<table>
<tr>
<td width="50%">
<h3>🎙️ Starter Agent</h3>
<p>A starter agent optimized for voice conversations.</p>
<p>
<a href="examples/voice_agents/basic_agent.py">Code</a>
</p>
</td>
<td width="50%">
<h3>🔄 Multi-user push to talk</h3>
<p>Responds to multiple users in the room via push-to-talk.</p>
<p>
<a href="examples/voice_agents/push_to_talk.py">Code</a>
</p>
</td>
</tr>

<tr>
<td width="50%">
<h3>🎵 Background audio</h3>
<p>Background ambient and thinking audio to improve realism.</p>
<p>
<a href="examples/voice_agents/background_audio.py">Code</a>
</p>
</td>
<td width="50%">
<h3>🛠️ Dynamic tool creation</h3>
<p>Creating function tools dynamically.</p>
<p>
<a href="examples/voice_agents/dynamic_tool_creation.py">Code</a>
</p>
</td>
</tr>

<tr>
<td width="50%">
<h3>☎️ Outbound caller</h3>
<p>Agent that makes outbound phone calls</p>
<p>
<a href="https://github.com/livekit-examples/outbound-caller-python">Code</a>
</p>
</td>
<td width="50%">
<h3>📋 Structured output</h3>
<p>Using structured output from LLM to guide TTS tone.</p>
<p>
<a href="examples/voice_agents/structured_output.py">Code</a>
</p>
</td>
</tr>

<tr>
<td width="50%">
<h3>🔌 MCP support</h3>
<p>Use tools from MCP servers</p>
<p>
<a href="examples/voice_agents/mcp">Code</a>
</p>
</td>
<td width="50%">
<h3>💬 Text-only agent</h3>
<p>Skip voice altogether and use the same code for text-only integrations</p>
<p>
<a href="examples/other/text_only.py">Code</a>
</p>
</td>
</tr>

<tr>
<td width="50%">
<h3>📝 Multi-user transcriber</h3>
<p>Produce transcriptions from all users in the room</p>
<p>
<a href="examples/other/transcription/multi-user-transcriber.py">Code</a>
</p>
</td>
<td width="50%">
<h3>🎥 Video avatars</h3>
<p>Add an AI avatar with Tavus, Beyond Presence, and Bithuman</p>
<p>
<a href="examples/avatar_agents/">Code</a>
</p>
</td>
</tr>

<tr>
<td width="50%">
<h3>🍽️ Restaurant ordering and reservations</h3>
<p>Full example of an agent that handles calls for a restaurant.</p>
<p>
<a href="examples/voice_agents/restaurant_agent.py">Code</a>
</p>
</td>
<td width="50%">
<h3>👁️ Gemini Live vision</h3>
<p>Full example (including iOS app) of Gemini Live agent that can see.</p>
<p>
<a href="https://github.com/livekit-examples/vision-demo">Code</a>
</p>
</td>
</tr>

</table>

## Running your agent

### Testing in terminal

```shell
python myagent.py console
```

Runs your agent in terminal mode, enabling local audio input and output for testing.
This mode doesn't require external servers or dependencies and is useful for quickly validating behavior.

### Developing with LiveKit clients

```shell
python myagent.py dev
```

Starts the agent server and enables hot reloading when files change. This mode allows each process to host multiple concurrent agents efficiently.

The agent connects to LiveKit Cloud or your self-hosted server. Set the following environment variables:
- LIVEKIT_URL
- LIVEKIT_API_KEY
- LIVEKIT_API_SECRET

You can connect using any LiveKit client SDK or telephony integration.
To get started quickly, try the [Agents Playground](https://agents-playground.livekit.io/).

### Running for production

```shell
python myagent.py start
```

Runs the agent with production-ready optimizations.

## Contributing

The Agents framework is under active development in a rapidly evolving field. We welcome and appreciate contributions of any kind, be it feedback, bugfixes, features, new plugins and tools, or better documentation. You can file issues under this repo, open a PR, or chat with us in LiveKit's [Slack community](https://livekit.io/join-slack).

<!--BEGIN_REPO_NAV-->
<br/><table>
<thead><tr><th colspan="2">LiveKit Ecosystem</th></tr></thead>
<tbody>
<tr><td>LiveKit SDKs</td><td><a href="https://github.com/livekit/client-sdk-js">Browser</a> · <a href="https://github.com/livekit/client-sdk-swift">iOS/macOS/visionOS</a> · <a href="https://github.com/livekit/client-sdk-android">Android</a> · <a href="https://github.com/livekit/client-sdk-flutter">Flutter</a> · <a href="https://github.com/livekit/client-sdk-react-native">React Native</a> · <a href="https://github.com/livekit/rust-sdks">Rust</a> · <a href="https://github.com/livekit/node-sdks">Node.js</a> · <a href="https://github.com/livekit/python-sdks">Python</a> · <a href="https://github.com/livekit/client-sdk-unity">Unity</a> · <a href="https://github.com/livekit/client-sdk-unity-web">Unity (WebGL)</a> · <a href="https://github.com/livekit/client-sdk-esp32">ESP32</a></td></tr><tr></tr>
<tr><td>Server APIs</td><td><a href="https://github.com/livekit/node-sdks">Node.js</a> · <a href="https://github.com/livekit/server-sdk-go">Golang</a> · <a href="https://github.com/livekit/server-sdk-ruby">Ruby</a> · <a href="https://github.com/livekit/server-sdk-kotlin">Java/Kotlin</a> · <a href="https://github.com/livekit/python-sdks">Python</a> · <a href="https://github.com/livekit/rust-sdks">Rust</a> · <a href="https://github.com/agence104/livekit-server-sdk-php">PHP (community)</a> · <a href="https://github.com/pabloFuente/livekit-server-sdk-dotnet">.NET (community)</a></td></tr><tr></tr>
<tr><td>UI Components</td><td><a href="https://github.com/livekit/components-js">React</a> · <a href="https://github.com/livekit/components-android">Android Compose</a> · <a href="https://github.com/livekit/components-swift">SwiftUI</a> · <a href="https://github.com/livekit/components-flutter">Flutter</a></td></tr><tr></tr>
<tr><td>Agents Frameworks</td><td><b>Python</b> · <a href="https://github.com/livekit/agents-js">Node.js</a> · <a href="https://github.com/livekit/agent-playground">Playground</a></td></tr><tr></tr>
<tr><td>Services</td><td><a href="https://github.com/livekit/livekit">LiveKit server</a> · <a href="https://github.com/livekit/egress">Egress</a> · <a href="https://github.com/livekit/ingress">Ingress</a> · <a href="https://github.com/livekit/sip">SIP</a></td></tr><tr></tr>
<tr><td>Resources</td><td><a href="https://docs.livekit.io">Docs</a> · <a href="https://github.com/livekit-examples">Example apps</a> · <a href="https://livekit.io/cloud">Cloud</a> · <a href="https://docs.livekit.io/home/self-hosting/deployment">Self-hosting</a> · <a href="https://github.com/livekit/livekit-cli">CLI</a></td></tr>
</tbody>
</table>
<!--END_REPO_NAV-->
