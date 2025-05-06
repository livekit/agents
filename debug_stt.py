import sys
import os

# Add the current directory to the path so we can import from livekit
sys.path.append(os.getcwd())

try:
    from livekit.agents.stt.stt import STT, STTCapabilities
    print("Successfully imported STT classes")
    
    # Create a test STT instance
    class TestSTT(STT):
        async def _recognize_impl(self, buffer, *, language=None, conn_options=None):
            # Implement the abstract method
            from livekit.agents.stt.stt import SpeechEvent, SpeechEventType, SpeechData
            return SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT,
                              alternatives=[SpeechData(language="en", text="test", confidence=1.0)])
    
    test_stt = TestSTT(capabilities=STTCapabilities(streaming=True, interim_results=True))
    print(f"STT type: {type(test_stt)}")
    print(f"Has capabilities: {hasattr(test_stt, 'capabilities')}")
    print(f"Capabilities: {test_stt.capabilities}")
    print(f"Capabilities streaming: {test_stt.capabilities.streaming}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()