# AWS Plugin for LiveKit Agents

Complete AWS AI integration for LiveKit Agents, including Bedrock, Polly, Transcribe, and realtime speech-to-speech support for Amazon Nova Sonic

**What's included:**
- **RealtimeModel** - Amazon Nova 2 Sonic and Nova Sonic 1.0 for speech-to-speech
- **LLM** - Powered by Amazon Bedrock, defaults to Nova 2 Lite
- **STT** - Powered by Amazon Transcribe
- **TTS** - Powered by Amazon Polly

See [https://docs.livekit.io/agents/integrations/aws/](https://docs.livekit.io/agents/integrations/aws/) for more information.

## ⚠️ Breaking Change

**Default model changed to Nova 2 Sonic**: `RealtimeModel()` now defaults to `amazon.nova-2-sonic-v1:0` with `modalities="mixed"` (was `amazon.nova-sonic-v1:0` with `modalities="audio"`).

If you need the previous behavior, explicitly specify Nova Sonic 1.0:
```python
model = aws.realtime.RealtimeModel.with_nova_sonic_1()
# or
model = aws.realtime.RealtimeModel(
    model="amazon.nova-sonic-v1:0",
    modalities="audio"
)
```

## Installation

```bash
pip install livekit-plugins-aws

# For Nova Sonic realtime models
pip install livekit-plugins-aws[realtime]
```

## Prerequisites

### AWS Credentials

You'll need AWS credentials with access to Amazon Bedrock. Set them as environment variables:

```bash
export AWS_ACCESS_KEY_ID=<your-access-key>
export AWS_SECRET_ACCESS_KEY=<your-secret-key>
export AWS_DEFAULT_REGION=us-east-1  # or your preferred region
```

### Getting Temporary Credentials from SSO (Local Testing)

If you use AWS SSO for authentication, get temporary credentials for local testing:

```bash
# Login to your SSO profile
aws sso login --profile your-profile-name

# Export credentials from your SSO session
eval $(aws configure export-credentials --profile your-profile-name --format env)

# Verify credentials are set
aws sts get-caller-identity
```

Alternatively, add this to your shell profile for automatic credential export:

```bash
# Add to ~/.bashrc or ~/.zshrc
function aws-creds() {
    eval $(aws configure export-credentials --profile $1 --format env)
}

# Usage: aws-creds your-profile-name
```

## Quick Start Example

The `realtime_joke_teller.py` example demonstrates both realtime and pipeline modes:

### Demonstrates Both Modes
- **Realtime mode**: Nova 2 Sonic for end-to-end speech-to-speech
- **Pipeline mode**: Amazon Transcribe + Nova 2 Lite + Amazon Polly

### Demonstrates Nova 2 Sonic Capabilities
- **Text prompting**: Agent greets users first using `generate_reply()`
- **Multilingual support**: Automatic language detection and response in 7 languages
- **Multiple voices**: 18 expressive voices across languages
- **Function calling**: Weather lookup, web search, and joke telling

### Setup

1. **Install dependencies:**
   ```bash
   pip install livekit-plugins-aws[realtime] \
               livekit-plugins-silero \
               jokeapi \
               duckduckgo-search \
               python-weather \
               python-dotenv
   ```

2. **Copy the example locally:**
   ```bash
   curl -O https://raw.githubusercontent.com/livekit/agents/main/examples/voice_agents/realtime_joke_teller.py
   ```

3. **Set up environment variables:**
   ```bash
   # Create .env file
   echo "AWS_DEFAULT_REGION=us-east-1" > .env
   # Add your AWS credentials (see Prerequisites above)
   ```

4. **(Optional) Run local LiveKit server:**
   
   For testing without LiveKit Cloud, run a local server:
   ```bash
   # Install LiveKit server
   brew install livekit  # macOS
   # or download from https://github.com/livekit/livekit/releases
   
   # Run in dev mode
   livekit-server --dev
   ```
   
   Add to your `.env` file:
   ```bash
   LIVEKIT_URL=wss://127.0.0.1:7880
   LIVEKIT_API_KEY=devkey
   LIVEKIT_API_SECRET=secret
   ```
   
   See [self-hosting documentation](https://docs.livekit.io/home/self-hosting/local/) for more details.

### Running the Example

**Realtime Mode (Nova 2 Sonic)** - Recommended for testing:
```bash
python realtime_joke_teller.py console
```
This runs locally using your computer's speakers and microphone. **Use a headset to prevent echo.**

**Multilingual Support:** Nova 2 Sonic automatically detects and responds in your language. Just start speaking in your preferred language (English, French, Italian, German, Spanish, Portuguese, or Hindi) and Nova 2 Sonic will respond in the same language!

**Pipeline Mode (Transcribe + Nova Lite + Polly)**:
```bash
python realtime_joke_teller.py console --mode pipeline
```

**Dev Mode** (connect to LiveKit room for remote testing):
```bash
python realtime_joke_teller.py dev
# or
python realtime_joke_teller.py dev --mode pipeline
```

Try asking:
- "What's the weather in Seattle?"
- "Tell me a programming joke"
- "Search for information about my favorite movie, Short Circuit"

## Features

### Nova 2 Sonic Capabilities

Amazon Nova 2 Sonic is a unified speech-to-speech foundation model that delivers:

- **Realtime bidirectional streaming** - Low-latency, natural conversations
- **Multilingual support** - English, French, Italian, German, Spanish, Portuguese, and Hindi
- **Automatic language mirroring** - Responds in the user's spoken language
- **Polyglot voices** - Matthew and Tiffany can seamlessly switch between languages within a single conversation, ideal for multilingual applications
- **18 expressive voices** - Multiple voices per language with natural prosody
- **Function calling** - Built-in tool use and agentic workflows
- **Interruption handling** - Graceful handling without losing context
- **Noise robustness** - Works in real-world environments
- **Text input support** - Programmatic text prompting

### Model Selection

```python
from livekit.plugins import aws

# Nova 2 Sonic (audio + text input, latest)
model = aws.realtime.RealtimeModel.with_nova_sonic_2()

# Nova Sonic 1.0 (audio-only, original model)
model = aws.realtime.RealtimeModel.with_nova_sonic_1()
```

### Voice Selection

Voices are specified as lowercase strings. Import `SONIC1_VOICES` or `SONIC2_VOICES` type hints for IDE autocomplete.

```python
from livekit.plugins.aws.experimental.realtime import SONIC2_VOICES

model = aws.realtime.RealtimeModel.with_nova_sonic_2(
    voice="carolina"  # Portuguese, feminine
)
```

#### Nova 2 Sonic Voice IDs (18 voices)

See [official documentation](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-language-support.html) for most up-to-date list and IDs.

- **English (US)**: `tiffany` (polyglot), `matthew` (polyglot)
- **English (UK)**: `amy`
- **English (Australia)**: `olivia`
- **English (India)**: `kiara`, `arjun`
- **French**: `ambre`, `florian`
- **Italian**: `beatrice`, `lorenzo`
- **German**: `tina`, `lennart`
- **Spanish (US)**: `lupe`, `carlos`
- **Portuguese (Brazil)**: `carolina`, `leo`
- **Hindi**: `kiara`, `arjun`

**Note**: Tiffany abd Matthew in Nova 2 Sonic support polyglot mode, seamlessly switching between languages within a single conversation.

#### Nova Sonic 1.0 Voice IDs (11 voices)

See [official documentation](https://docs.aws.amazon.com/nova/latest/userguide/available-voices.html) for most up-to-date list and IDs.

- **English (US)**: `tiffany`, `matthew`
- **English (UK)**: `amy`
- **French**: `ambre`, `florian`
- **Italian**: `beatrice`, `lorenzo`
- **German**: `greta`, `lennart`
- **Spanish**: `lupe`, `carlos`

### Text Prompting with `generate_reply()`

Nova 2 Sonic supports programmatic text input. This can be used to trigger agent responses or to mix speech and text input within a UI in the same conversation:

```python
class Assistant(Agent):
    async def on_enter(self):
        # Make the agent speak first with a greeting
        await self.session.generate_reply(
            instructions="Greet the user and introduce your capabilities"
        )
```

#### `instructions` vs `user_input`

The `generate_reply()` method accepts two parameters with different behaviors:

**`instructions`** - System-level commands (recommended):
```python
await session.generate_reply(
    instructions="Greet the user warmly and ask how you can help"
)
```
- Sent as a system prompt/command to the model
- Triggers immediate generation
- Does not appear in conversation history as user message
- Use for: Agent-initiated speech, prompting specific behaviors

**`user_input`** - Simulated user messages:
```python
await session.generate_reply(
    user_input="Hello, I need help with my account"
)
```
- Sent as interactive USER role content
- Added to Nova's conversation context
- Triggers generation as if user spoke
- Use for: Testing, simulating user input, programmatic conversations

**When to use each:**
- **Agent greetings**: Use `instructions` - agent should speak without user input
- **Guided responses**: Use `instructions` - direct the agent's next action
- **Simulated conversations**: Use `user_input` - test multi-turn dialogs
- **Programmatic user input**: Use `user_input` - inject text as if user spoke

### Turn-Taking Sensitivity

Control how quickly the agent responds to pauses:

```python
model = aws.realtime.RealtimeModel.with_nova_sonic_2(
    turn_detection="MEDIUM"  # HIGH, MEDIUM (default), LOW
)
```

- **HIGH**: Fastest response time, optimized for latency. May interrupt slower speakers
- **MEDIUM**: Balanced approach with moderate response time. Reduces false positives while maintaining responsiveness (recommended)
- **LOW**: Slowest response time with maximum patience, better for hesitant speakers

### Complete Example

```python
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import aws
from dotenv import load_dotenv


load_dotenv()

class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant powered by Amazon Nova 2 Sonic."
        )
    
    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Greet the user and offer assistance"
        )

server = agents.AgentServer()

@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    session = AgentSession(
        llm=aws.realtime.RealtimeModel.with_nova_sonic_2(
            voice="matthew",
            turn_detection="MEDIUM",
            tool_choice="auto"
        )
    )
    
    await session.start(room=ctx.room, agent=Assistant())

if __name__ == "__main__":
    agents.cli.run_app(server)
```

## Pipeline Mode (STT + LLM + TTS)

For more control over individual components, use pipeline mode:

```python
from livekit.plugins import aws, silero

session = AgentSession(
    stt=aws.STT(),                    # Amazon Transcribe
    llm=aws.LLM(),                    # Nova 2 Lite (default)
    tts=aws.TTS(),                    # Amazon Polly
    vad=silero.VAD.load(),
)
```

### Nova 2 Lite

Amazon Nova 2 Lite is a fast, cost-effective reasoning model optimized for everyday AI workloads:

- **Lightning-fast processing** - Very low latency for real-time conversations
- **Cost-effective** - Industry-leading price-performance
- **Multimodal inputs** - Text, image, and video ([documentation](https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-multimodal-models.html))
- **1 million token context window** - Handle long conversations and complex context ([source](https://aws.amazon.com/blogs/aws/introducing-amazon-nova-2-lite-a-fast-cost-effective-reasoning-model/))
- **Agentic workflows** - RAG systems, function calling, tool use
- **Fine-tuning support** - Customize for your specific use case

Ideal for pipeline mode where you need fast, accurate LLM responses in voice applications.

## Resources

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [Amazon Nova Documentation](https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-conversational-speech.html)
- [Example: realtime_joke_teller.py](https://github.com/livekit/agents/blob/main/examples/voice_agents/realtime_joke_teller.py)
