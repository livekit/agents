# Resemble plugin for LiveKit Agents

Support for voice synthesis with the [Resemble AI](https://www.resemble.ai/) API, using both their REST API and WebSocket streaming interface.

See [https://docs.livekit.io/agents/integrations/tts/resemble/](https://docs.livekit.io/agents/integrations/tts/resemble/) for more information.

## Installation

```bash
pip install livekit-plugins-resemble
```

## Pre-requisites

You'll need an API key from Resemble AI. It can be set as an environment variable: `RESEMBLE_API_KEY`

Additionally, you'll need the voice UUID from your Resemble AI account.

## Examples

### Recommended

```python
import asyncio
from livekit.plugins.resemble import TTS

async def run_tts_example():
    # Use TTS with async context manager for automatic resource cleanup
    async with TTS(
        api_key="your_api_key",  # or set RESEMBLE_API_KEY environment variable
        voice_uuid="your_voice_uuid",
        # Optional parameters
        sample_rate=44100,  # Sample rate in Hz (default: 44100)
        precision="PCM_16",  # Audio precision (PCM_32, PCM_24, PCM_16, MULAW)
        output_format="wav",  # Output format (wav or mp3)
    ) as tts:
        # One-off synthesis (uses REST API)
        audio_stream = tts.synthesize("Hello, world!")
        
        # Process chunks as they arrive
        async for chunk in audio_stream:
            # Audio data is in the 'frame.data' attribute of SynthesizedAudio objects
            audio_data = chunk.frame.data
            print(f"Received chunk: {len(audio_data)} bytes")
        
        # Alternative: collect all audio at once into a single AudioFrame
        audio_stream = tts.synthesize("Another example sentence.")
        audio_frame = await audio_stream.collect()
        print(f"Collected complete audio: {len(audio_frame.data)} bytes")
        
        # Real-time streaming synthesis (uses WebSocket API)
        # Only available for Business plan users in Resemble AI
        stream = tts.stream()
        await stream.synthesize_text("Hello, world!")
        


# Run the example
asyncio.run(run_tts_example())
```

### Alternative: Manual Resource Management

If you prefer to manage resources manually, make sure to properly clean up:

```python
import asyncio
from livekit.plugins.resemble import TTS

async def run_tts_example():
    # Initialize TTS with your credentials
    tts = TTS(
        api_key="your_api_key", 
        voice_uuid="your_voice_uuid",
    )

    try:
        # TTS operations
        audio_stream = tts.synthesize("Hello, world!")
        async for chunk in audio_stream:
            # Access audio data correctly
            process_audio(chunk.frame.data)
    finally:
        # Always clean up resources when done
        await tts.aclose()

# Run the example
asyncio.run(run_tts_example())
```

### Resource Management

When using this plugin outside of the LiveKit agent framework, it's important to properly manage the TTS instance lifecycle:

1. **Preferred method**: Use the async context manager pattern (`async with TTS(...) as tts:`)
2. If managing manually, always call `await tts.aclose()` in a finally block
3. If you prefer to provide your own HTTP session, you can pass it using the `http_session` parameter:

```python
import aiohttp

async def with_custom_session():
    async with aiohttp.ClientSession() as session:
        async with TTS(
            api_key="your_api_key",
            voice_uuid="your_voice_uuid",
            http_session=session
        ) as tts:
            # Use TTS...
            # No need to manually close anything - context managers handle it all
```

## Implementation Details

This plugin uses two different approaches to generate speech:

1. **One-off Synthesis** - Uses Resemble's REST API for simple text-to-speech conversion
2. **Streaming Synthesis** - Uses Resemble's WebSocket API for real-time streaming synthesis

The WebSocket streaming API is only available for Resemble AI Business plan users. 