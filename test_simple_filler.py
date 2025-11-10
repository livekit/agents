# test_simple_filler.py
# Standalone loader: loads filler_handler.py by path and tests it
import importlib.util
import asyncio
import pathlib
import sys

# path to your filler handler file
fh_path = pathlib.Path("livekit-agents") / "livekit" / "agents" / "filler_handler.py"

spec = importlib.util.spec_from_file_location("filler_handler_module", str(fh_path))
mod = importlib.util.module_from_spec(spec)
sys.modules["filler_handler_module"] = mod
spec.loader.exec_module(mod)

# now use the FillerHandler class
FillerHandler = mod.FillerHandler

async def run_test():
    handler = FillerHandler()
    handler.agent_speaking = True
    print(await handler.handle_transcription("umm", 0.9))       # expect: ignore
    print(await handler.handle_transcription("wait stop", 0.9)) # expect: interrupt
    handler.agent_speaking = False
    print(await handler.handle_transcription("umm", 0.95))      # expect: register

if __name__ == "__main__":
    asyncio.run(run_test())
