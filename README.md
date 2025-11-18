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

# Filler Word Filter Extension for LiveKit Agents

## Overview

This extension adds intelligent filler word filtering to LiveKit voice agents, preventing false interruptions from common filler words like "umm", "uh", "hmm", etc.

## Features

- ✅ **Stateful Filtering**: Different behavior when agent is speaking vs. quiet
- ✅ **Configurable**: Customizable list of ignored filler words
- ✅ **Real-time**: Minimal latency overhead
- ✅ **Confidence Filtering**: Ignores low-confidence ASR results
- ✅ **Dynamic Updates**: Update ignored words at runtime
- ✅ **Multi-language Ready**: Supports filler words from any language
- ✅ **Comprehensive Logging**: Detailed logs for debugging

## How It Works

### State-Based Logic

1. **When Agent is QUIET** (not speaking):

   - All user speech is processed immediately
   - Even filler words like "umm" start the user's turn

2. **When Agent is SPEAKING**:
   - Filler-only phrases (e.g., "umm", "uh hmm") are ignored
   - Real words trigger immediate interruption (e.g., "stop", "wait")
   - Mixed phrases interrupt (e.g., "umm okay stop")

### Confidence Filtering

Transcripts with confidence scores below the threshold (default: 0.3) are automatically ignored to reduce false positives from background noise.

## Installation

```bash
pip install livekit-agents livekit-plugins-openai livekit-plugins-silero
```

## Quick Start

### Basic Usage

```python
from livekit.agents import JobContext, cli, WorkerOptions
from src.filler_filter import create_assistant_with_filter

async def entrypoint(ctx: JobContext):
    # Create assistant with default filler filtering
    assistant, filter_ext = create_assistant_with_filter(ctx)

    await ctx.connect()
    assistant.start(ctx.room)
    await assistant.say("Hello! I'm ready to chat.")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### Custom Configuration

```python
from src.filler_filter import create_assistant_with_filter, FillerFilterConfig

# Create custom config
config = FillerFilterConfig()
config.update_ignored_words(['uh', 'umm', 'hmm', 'haan', 'toh', 'accha'])

# Create assistant with custom config
assistant, filter_ext = create_assistant_with_filter(ctx, config)
```

### Environment Variable Configuration

Set the `IGNORED_FILLER_WORDS` environment variable:

```bash
# Comma-separated list
export IGNORED_FILLER_WORDS="uh,umm,hmm,haan,toh"

# Or JSON array
export IGNORED_FILLER_WORDS='["uh","umm","hmm","haan"]'
```

### Dynamic Updates at Runtime

```python
# Update the list of ignored words while running
filter_ext.update_ignored_words(['uh', 'umm', 'hmm', 'well'])
```

## Testing

### Run Unit Tests

```bash
python -m pytest tests/test_filler_filter.py -v
```

### Manual Testing Scenarios

#### Test Case 1: Filler-Only Interruption (Should Be Ignored)

1. Start the agent
2. Let it speak a long response
3. Say "umm" or "uh" while it's talking
4. **Expected**: Agent continues speaking
5. **Check logs**: Should see "Ignored filler interruption"

#### Test Case 2: Real Interruption (Should Stop Agent)

1. Start the agent
2. Let it speak
3. Say "wait" or "stop" while it's talking
4. **Expected**: Agent stops immediately
5. **Check logs**: Should see "Detected real interruption"

#### Test Case 3: Mixed Interruption (Should Stop Agent)

1. Start the agent
2. Let it speak
3. Say "umm okay stop" while it's talking
4. **Expected**: Agent stops immediately
5. **Check logs**: Should see "Detected real interruption" with real words: ['okay', 'stop']

#### Test Case 4: Filler When Agent is Quiet (Should Process)

1. Wait for agent to finish speaking
2. Say "umm"
3. **Expected**: Agent recognizes this as the start of your turn
4. **Check logs**: Should see "Agent quiet - processing speech"

### Performance Testing

```python
import time
from src.filler_filter import FillerWordFilter

filter = FillerWordFilter()
filter.set_agent_speaking_state(True)

start = time.perf_counter()
for _ in range(1000):
    filter.should_process_interruption("umm okay stop")
end = time.perf_counter()

print(f"Average processing time: {(end-start)/1000*1000:.3f}ms")
# Expected: < 1ms per call
```

## Configuration Reference

### FillerFilterConfig

| Parameter        | Type     | Default   | Description                      |
| ---------------- | -------- | --------- | -------------------------------- |
| `ignored_words`  | Set[str] | See below | Set of words to ignore           |
| `min_confidence` | float    | 0.3       | Minimum ASR confidence threshold |

**Default Ignored Words**: `['uh', 'umm', 'hmm', 'haan', 'um', 'er', 'ah', 'eh']`

### Methods

- `update_ignored_words(words: List[str])`: Replace entire list
- `add_ignored_word(word: str)`: Add single word
- `remove_ignored_word(word: str)`: Remove single word

## Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Log messages include:

- `INFO: Agent quiet - processing speech: 'hello'`
- `INFO: Detected real interruption: 'stop' (real words: ['stop'])`
- `INFO: Ignored filler interruption: 'umm' (only fillers: ['umm'])`
- `INFO: Ignored low-confidence (0.25) transcript: 'noise'`

## Troubleshooting

### Agent not stopping on real words

- Check that `is_agent_speaking` state is being set correctly
- Verify the word isn't in the `ignored_words` list
- Enable DEBUG logging to see filtering decisions

### Agent stopping on fillers

- Verify the filler is in the `ignored_words` list (case-insensitive)
- Check for punctuation issues (e.g., "umm." vs "umm")
- Review ASR confidence scores

### High latency

- The filter adds < 1ms overhead per transcript
- Check your ASR service latency instead
- Consider adjusting VAD sensitivity

## Architecture
