"""Attachment helper for the LiveKit interruption handler extension.

This module provides a convenience function to attach the InterruptHandler
to an AgentSession. Wiring logic will be added in later steps.
"""

from .interrupt_handler import InterruptHandler


def attach_interrupt_handler(session, config=None):
    """Attach interrupt handler to an AgentSession. Logic added later.

    This function constructs and returns a new InterruptHandler instance,
    and wires it into the AgentSession event stream using public callbacks.
    """

    handler = InterruptHandler(session, config=config)

    if hasattr(session, "on"):
        session.on("agent_state_changed", handler.on_agent_state_changed)
        session.on("user_input_transcribed", handler.on_user_input_transcribed)

    return handler
