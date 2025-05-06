# LiveKit Plugins Edge TTS

Agent Framework plugin for text-to-speech using [Microsoft Edge TTS](https://github.com/rany2/edge-tts).

## Installation

```bash
pip install livekit-plugins-edge-tts
```

## Usage

```python
import asyncio
from livekit.plugins.edge_tts import TTS, VoiceSettings

async def main():
    # Create a TTS instance with default settings
    tts_engine = TTS()
    
    # Or customize with specific voice and settings
    tts_engine = TTS(
        voice="en-US-AriaNeural",  # Female voice
        rate="+10%",               # Speak 10% faster
        volume="+20%",             # 20% louder
        pitch="-5%",               # Slightly lower pitch
    )
    
    # List available voices
    voices = await TTS.list_voices()
    print(f"Available voices: {len(voices)}")
    
    # Synthesize text to speech
    stream = tts_engine.synthesize("Hello, this is a test of the Edge TTS plugin for LiveKit.")
    
    # Process the audio stream
    async for event in stream:
        # Process audio frames
        pass
    
    # Clean up
    await tts_engine.aclose()

if __name__ == "__main__":
    asyncio.run(main())
```

## Available Voices

Edge TTS provides a wide range of voices across multiple languages. You can list all available voices using the `TTS.list_voices()` method.

Some common voices include:
- `en-US-ChristopherNeural` (Male)
- `en-US-AriaNeural` (Female)
- `en-GB-RyanNeural` (Male)
- `en-GB-SoniaNeural` (Female)
- `fr-FR-HenriNeural` (Male)
- `fr-FR-DeniseNeural` (Female)
- `de-DE-ConradNeural` (Male)
- `de-DE-KatjaNeural` (Female)
- `es-ES-AlvaroNeural` (Male)
- `es-ES-ElviraNeural` (Female)
- `ja-JP-KeitaNeural` (Male)
- `ja-JP-NanamiNeural` (Female)

## Voice Settings

You can adjust the following voice settings:

- `rate`: Speaking rate adjustment (e.g., "+0%", "+10%", "-10%")
- `volume`: Volume adjustment (e.g., "+0%", "+10%", "-10%")
- `pitch`: Pitch adjustment (e.g., "+0%", "+10%", "-10%")

## Requirements

- Python 3.9+
- edge-tts 6.1.9+
- livekit-agents 1.0.18+