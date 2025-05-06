import sys
import os

# Add the current directory to the path so we can import from livekit
sys.path.append(os.getcwd())

try:
    # Import the core STT classes
    from livekit.agents.stt.stt import STT, STTCapabilities
    print("Successfully imported core STT classes")
    
    # Try to import the Whisper plugin
    try:
        from livekit.plugins.whisper.stt import main
        print("Found Whisper STT module, but it doesn't seem to implement the STT interface")
    except ImportError as e:
        print(f"Error importing Whisper STT: {e}")
    
    # Check if there's any class in the Whisper module that inherits from STT
    import inspect
    import pkgutil
    
    # List all modules in livekit.plugins.whisper
    print("\nModules in livekit.plugins.whisper:")
    import livekit.plugins.whisper
    for _, name, _ in pkgutil.iter_modules(livekit.plugins.whisper.__path__, livekit.plugins.whisper.__name__ + "."):
        print(f"  {name}")
    
    # Check if any module has a class that inherits from STT
    print("\nChecking for STT implementations:")
    for _, name, _ in pkgutil.iter_modules(livekit.plugins.whisper.__path__, livekit.plugins.whisper.__name__ + "."):
        try:
            module = __import__(name, fromlist=["*"])
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, STT) and obj != STT:
                    print(f"  Found STT implementation: {obj.__name__} in {module.__name__}")
        except Exception as e:
            print(f"  Error inspecting {name}: {e}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()