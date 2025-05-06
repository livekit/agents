import sys
import os

# Add the current directory to the path so we can import from livekit
sys.path.append(os.getcwd())

try:
    # Import the core STT classes
    from livekit.agents.stt.stt import STT, STTCapabilities
    print("Successfully imported core STT classes")
    
    # Import our new WhisperSTT class
    from livekit.plugins.whisper import WhisperSTT
    print("Successfully imported WhisperSTT")
    
    # Create an instance of WhisperSTT
    whisper_stt = WhisperSTT(model_name="openai/whisper-tiny")
    
    # Check if it has the capabilities attribute
    print(f"WhisperSTT type: {type(whisper_stt)}")
    print(f"Is instance of STT: {isinstance(whisper_stt, STT)}")
    print(f"Has capabilities: {hasattr(whisper_stt, 'capabilities')}")
    print(f"Capabilities: {whisper_stt.capabilities}")
    print(f"Capabilities streaming: {whisper_stt.capabilities.streaming}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()