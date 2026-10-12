# Planned: agents, tools and toolsets

Status: designed, not built. Extends [agents.md](agents.md) and
[tools.md](tools.md).

## Streaming runs

`run_stream()` is `run()` with tokens as they arrive, and `iter()` walks the
same loop step by step. Both take the arguments of `run()`.

## Toolsets on an agent

```python
from livekit.agents.llm import FilteredToolset, PrefixedToolset

def verified_only(ctx: RunContext[Deps, Member], tool: Tool) -> bool:
    return ctx.state.verified or tool.id == "block_card"

member_services = Agent(
    ...,
    toolsets=[
        FilteredToolset(Cards(), verified_only),     # chosen per turn, from the state
        PrefixedToolset(rewards, "rewards_"),        # rewards_balance, rewards_redeem
    ],
)
```

| Class | What |
|---|---|
| `Agent(toolsets=)` | toolsets attached to the agent, next to `tools=` |
| `FilteredToolset(toolset, predicate)` | the predicate runs per turn with the `RunContext` and decides which tools the model sees |
| `PrefixedToolset(toolset, prefix)` | avoids name clashes between toolsets |

`Cards` and `rewards` are the `Toolset` subclass and the `MCPToolset` of
[tools.md](tools.md). A filtered toolset is how a tool appears once the
state allows it: the agent itself never changes.

## Hooks around the model

| Parameter or hook | When |
|---|---|
| `history_processors=[...]` | functions over the history before each model request |
| `@agent.before_model_call`, `@agent.after_model_call` | around each model request |
| `prepare_tools` | choose the tools for this turn |

## Capabilities

`capabilities=[...]` on the agent attaches bundles of hooks and tools.
`Tasks()` is the default capability that lets an agent see and steer its
background tasks; see [planned-agent-tasks.md](planned-agent-tasks.md).
`capabilities=[]` removes them.

See `examples/use-cases/planned-06_card_servicing_toolsets.py`.
