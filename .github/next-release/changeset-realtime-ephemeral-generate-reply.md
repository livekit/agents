---
"livekit-agents": patch
"livekit-plugins-openai": patch
---

Add `add_to_chat_ctx: bool = True` parameter to `RealtimeSession.generate_reply` and `AgentSession.generate_reply`, plus a `RealtimeCapabilities.ephemeral_response` capability flag plugins use to declare honor of the parameter. When `False` against a plugin that declares the capability (currently the OpenAI plugin on the public endpoint), the rendered assistant turn does not enter the substrate's persistent conversation state and is not written to the agent's local chat context. Plugins that do not declare the capability emit a `UserWarning` and fall back to the legacy add-to-context path. Default `add_to_chat_ctx=True` preserves all existing behavior.

The OpenAI plugin enforces a single-isolated-call serialization contract: a second `generate_reply(add_to_chat_ctx=False)` issued while the first is in flight raises `RuntimeError` with diagnostic context (`client_event_id`, `response_id`, elapsed-since-issue, docstring §Concurrency reference). Default `add_to_chat_ctx=True` calls retain their existing concurrency semantics.

Includes concurrency hardening for the substrate's parallel out-of-band code path (orphan filter at `response.created` and nine bare-assert handler conversions to early-return on `_current_generation is None`) and wires `interrupt()` with the active server-assigned `response_id` so cancel actually stops in-flight isolated responses.
