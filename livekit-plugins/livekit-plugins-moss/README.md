# Moss plugin for LiveKit Agents

This package wires the [Moss](https://www.moss.dev/) semantic search
SDK into the LiveKit Agents plugin ecosystem. It allows your agents to perform fast, semantic lookups across your custom knowledge base (FAQs, documentation, product specs) to ground their responses in real-time.

## Installation

```bash
cd livekit-plugins/livekit-plugins-moss
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools build
python -m pip install -e .
```

Set your credentials via environment variables:

```bash
export MOSS_PROJECT_ID="your-project-id"
export MOSS_PROJECT_KEY="your-project-key"
export MOSS_INDEX_NAME="demo"
```

## Usage

### Step 1: Create the Index

First, create a script to define your documentation and populate the Moss index.

```python
import asyncio
import os
from dotenv import load_dotenv
from livekit.plugins.moss import MossClient, DocumentInfo

load_dotenv()

async def setup() -> None:
    client = MossClient()
    index_name = os.environ.get("MOSS_INDEX_NAME", "demo")

    docs = [
        DocumentInfo(id="shipping-policy", text="Standard shipping takes 3-5 days."),
        DocumentInfo(id="return-policy", text="We offer a 7-day return policy."),
        DocumentInfo(id="payment-methods", text="We accept Credit Cards and PayPal.")
    ]

    await client.create_index(index_name, documents=docs)
    print(f"Index '{index_name}' created successfully.")

if __name__ == "__main__":
    asyncio.run(setup())
```

### Step 2: Run the Agent

Now, run the agent to query the index you just created. The LLM will use the `search_faqs` tool to find answers.

```python
import os
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, function_tool
from livekit.plugins import openai, deepgram, silero, cartesia
from livekit.plugins.moss import MossClient

load_dotenv()

class MossAssistant(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful assistant. Use 'search_faqs' to answer questions.")
        self._moss_client = MossClient()
        self._index_name = os.environ.get("MOSS_INDEX_NAME", "demo")

    async def on_enter(self) -> None:
        # Pre-load the index for fast performance
        await self._moss_client.load_index(self._index_name)
        await super().on_enter()

    @function_tool
    async def search_faqs(self, query: str) -> str:
        """Search the FAQ database for relevant information."""
        results = await self._moss_client.query(self._index_name, query)
        return "\n\n".join([doc.text for doc in results.docs]) or "No info found."

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session =  AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=openai.LLM(
            model="gpt-5",
        ),
        tts=cartesia.TTS(
            model="sonic-2",
            voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
        ),
        vad=silero.VAD.load(),
    )

    await session.start(room=ctx.room, agent=MossAssistant())
    await session.generate_reply(instructions="Greet the user warmly.")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

To test locally:

```bash
python my_agent.py console
```
