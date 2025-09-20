#!/usr/bin/env python3
import sys
print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nTrying to import livekit...")
try:
    from livekit.agents import Agent, AgentSession
    from livekit.agents.stt import deepgram
    from livekit.agents.tts import cartesia
    from livekit.agents.vad import silero
    print("✅ All imports successful from root directory!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()