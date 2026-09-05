from typing import Literal

# The Grid addresses capability tiers rather than a specific lab's model name;
# a tier routes to a current model for that tier. The `*-latest` ids pin a
# particular lab instead. `GET https://api.thegrid.ai/v1/models` is the
# authoritative list.
TheGridChatModels = Literal[
    "text-standard",
    "text-prime",
    "text-max",
    "code-standard",
    "code-prime",
    "code-max",
    "agent-standard",
    "agent-prime",
    "agent-max",
    "bytedance-pro-latest",
    "claude-opus-latest",
    "deepseek-pro-latest",
    "gemini-pro-latest",
    "glm-latest",
    "gpt-sol-latest",
    "kimi-latest",
    "minimax-latest",
]
