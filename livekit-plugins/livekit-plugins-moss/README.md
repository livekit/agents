# Moss plugin for LiveKit Agents

Knowledge base search with [Moss](https://www.moss.dev). The plugin loads a Moss index into the
agent's memory when the agent starts and runs hybrid (semantic and keyword) search in process, so
a search takes a few milliseconds and makes no network request.

## Installation

```bash
pip install livekit-plugins-moss
```

## Pre-requisites

Create an index in the [Moss portal](https://portal.usemoss.dev) or with the `moss` SDK. Set
`MOSS_PROJECT_ID` and `MOSS_PROJECT_KEY`, or pass `project_id` and `project_key`.

## Usage

As a tool the LLM calls when it needs facts:

```python
from livekit.agents import Agent
from livekit.plugins import moss

agent = Agent(
    instructions="You answer questions about our store. Use search_knowledge_base for facts.",
    tools=[moss.KnowledgeBase("support-faq")],
)
```

To search several indexes with the same tool, pass a list, such as
`moss.KnowledgeBase(["support-faq", "policies"])`. They must use the same embedding model.

Or also search on every user turn and add the passages to that turn, so the LLM can answer without
a tool call:

```python
class Assistant(Agent):
    def __init__(self) -> None:
        self.kb = moss.KnowledgeBase("support-faq")
        super().__init__(instructions="You answer questions about our store.", tools=[self.kb])

    async def llm_node(self, chat_ctx, tools, model_settings):
        await self.kb.add_context(chat_ctx)
        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)
```

Searching in `llm_node` works with preemptive generation and text input. Changes made in
`on_user_turn_completed` would discard the preemptive reply. Realtime models skip `llm_node`, so
give them the tool.

The passages from `add_context` reach the LLM next to your instructions, so use it with indexes
whose text you control, and rely on the tool for crawled or user-written content.

With several agents in one session, pass the knowledge base to `AgentSession(tools=[kb])` instead,
so a handoff doesn't close it and load the index again.
