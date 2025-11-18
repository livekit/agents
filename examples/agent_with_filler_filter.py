"""
Example Agent with Filler Filter Integration
"""
import os
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import deepgram, elevenlabs, openai, silero

# Import the filler filter components
from livekit_plugins.filler_filter import FillerFilterWrapper, AgentStateTracker, get_config


@function_tool
async def lookup_weather(context, location: str):
    """Used to look up weather information."""
    return {"weather": "sunny", "temperature": 70}


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent"""
    await ctx.connect()
    
    # Step 1: Create the agent state tracker
    state_tracker = AgentStateTracker()
    
    # Step 2: Create the base STT (any provider works)
    base_stt = deepgram.STT(model="nova-3")
    
    # Step 3: Wrap the STT with FillerFilterWrapper
    filtered_stt = FillerFilterWrapper(
        stt=base_stt,
        state_tracker=state_tracker
    )
    
    # Step 4: Create the agent
    agent = Agent(
        instructions="You are a friendly voice assistant built by LiveKit.",
        tools=[lookup_weather],
    )
    
    # Step 5: Create the session with filtered STT
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=filtered_stt,  # Use the wrapped STT
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(),
    )
    
    # Step 6: Attach state tracker to session (CRITICAL!)
    state_tracker.attach_to_session(session)
    
    # Step 7: Start the session
    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="greet the user and ask about their day")
    
    # Optional: Print statistics at the end
    @session.on("close")
    def on_close(event):
        stats = filtered_stt.get_stats()
        print(f"\n=== Filler Filter Statistics ===")
        print(f"Total events: {stats['total_events']}")
        print(f"Filtered: {stats['filtered_events']}")
        print(f"Passed: {stats['passed_events']}")
        if stats['total_events'] > 0:
            filter_rate = (stats['filtered_events'] / stats['total_events']) * 100
            print(f"Filter rate: {filter_rate:.1f}%")


if __name__ == "__main__":
    # Optional: Configure via environment variables
    # os.environ['FILLER_WORDS'] = '["uh","um","hmm","haan"]'
    # os.environ['FILLER_CONFIDENCE_THRESHOLD'] = '0.3'
    # os.environ['FILLER_DEBUG'] = 'true'
    
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
