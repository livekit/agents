import anthropic

# We can define up to 4 cache breakpoints, we will add them at:
# - the last tool definition
# - the last system message
# - the last assistant message
# - the last user message before the last assistant message
# https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#structuring-your-prompt
CACHE_CONTROL_EPHEMERAL = anthropic.types.CacheControlEphemeralParam(type="ephemeral")

__all__ = ["CACHE_CONTROL_EPHEMERAL"]
