from typing import Literal

# Model names accepted by Kokoro-FastAPI; all of them map to the same
# underlying Kokoro model on the server.
TTSModels = Literal["kokoro", "tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
