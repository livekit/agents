from ... import StopResponse, function_tool
from ...voice.events import RunContext


@function_tool
async def speak_to_ivr(ctx: RunContext, text: str) -> None:
    """
    Speak a specific phrase to the IVR using voice.

    Use when the IVR asks you to say something verbally rather than press keys —
    e.g., your name, "yes", "no", or any spoken menu option.
    After speaking, yields the turn back to the IVR to listen for its reply.
    """
    handle = ctx.session.say(text, allow_interruptions=False)
    await ctx.wait_for_playout()
    await handle
    raise StopResponse()
