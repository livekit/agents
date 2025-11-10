# tests/test_interrupt_filter.py (add these lines at top)
import sys, os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# repo_root now points to D:\salecode\agents
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)



import asyncio
from src.livekit_interrupt_filter import InterruptFilter

async def run_tests():
    f = InterruptFilter(ignored_words=["uh","umm","hmm"])
    
    r1 = await f.handle_transcription_event("uh", 0.95, True, agent_speaking=True)
    assert r1['action'] == 'ignore'
    
    r2 = await f.handle_transcription_event("umm", 0.9, True, agent_speaking=False)
    assert r2['action'] == 'register'
    
    r3 = await f.handle_transcription_event("umm stop", 0.9, True, agent_speaking=True)
    assert r3['action'] == 'stop'
   
    r4 = await f.handle_transcription_event("I need help", 0.95, True, agent_speaking=True)
    assert r4['action'] == 'stop'
    print("All tests passed.")

if __name__ == "__main__":
    asyncio.run(run_tests())
