# Agent iteration and toolsets

## Stepping a run

`iter()` walks the loop of `run()` step by step, with the arguments of
`run()`. Stream the reply with `agent.stream()` in
[agents.md](agents.md).

## Toolsets on an agent

```python
from livekit.agents.llm import FilteredToolset, PrefixedToolset

def verified_only(ctx: RunContext[MemberContext], tool: Tool) -> bool:
    return ctx.context.verified or tool.name == "block_card"

member_services = Agent(
    ...,
    tools=[
        FilteredToolset(Cards(), verified_only),     # chosen per turn, from the state
        PrefixedToolset(rewards, "rewards_"),        # rewards_balance, rewards_redeem
    ],
)
```

| Class | What |
|---|---|
| `Agent(tools=[...])` | tools and toolsets attached to the agent |
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
Each capability declares the hooks and tools it contributes.

See `examples/use-cases/planned-06_card_servicing_toolsets.py`.
