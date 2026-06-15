# Resemble plugin for LiveKit Agents

Support for [Resemble AI](https://www.resemble.ai/) voice synthesis and real-time
deepfake detection in LiveKit Agents.

See [https://docs.livekit.io/agents/integrations/tts/resemble/](https://docs.livekit.io/agents/integrations/tts/resemble/) for more information.

## Installation

```bash
pip install livekit-plugins-resemble
```

## Pre-requisites

You'll need an API key from Resemble AI. It can be set as an environment variable:
`RESEMBLE_API_KEY`.

For TTS, you'll also need the voice UUID from your Resemble AI account.

## Examples

### Text-to-speech

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
        model="chatterbox-turbo",
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

### Deepfake detection

Add Resemble Detect to a LiveKit room with a small, event-driven surface:

```python
from livekit.plugins import resemble

detect = resemble.ResembleDetect(security="standard")
detect.attach(ctx.room)  # auto-subscribes to the first remote microphone track


def on_synthetic(result: resemble.DetectionResult) -> None:
    # result.label is the raw Detect label ("fake"); normalized_label is app-facing.
    if result.normalized_label == "synthetic":
        # pause account actions, ask for step-up verification, or escalate
        ...


detect.on("synthetic_detected", on_synthetic)
```

Each result exposes a stable app payload:

```python
{
    "label": "synthetic",
    "score": 0.86,
    "confidence": 0.9,
    "window_ts": 41.2,
    "scan_index": 3,
    "is_final": False,
}
```

Use `result.to_dict()` or `verdict.to_dict()` if you want this shape directly.

### Detection options

1. **Standard security** - default for most calls. Checks a 4s speech window early, samples
   across the call, and emits `synthetic_detected` only after 2-of-3 recent checks agree.

   ```python
   detect = resemble.ResembleDetect(security="standard")
   ```

2. **Spot check** - lowest cost. Runs one check once enough speech is available.

   ```python
   detect = resemble.ResembleDetect(security="spot")
   ```

3. **High security** - continuous monitoring for sensitive workflows.

   ```python
   detect = resemble.ResembleDetect(security="high")
   ```

4. **Custom policy** - override any preset with simple keyword arguments.

   ```python
   detect = resemble.ResembleDetect(
       security="standard",
       window_seconds=4.0,
       sample_interval_seconds=20.0,
       fake_threshold=0.75,
       agreement_window=3,
       min_fake_results=2,
       zero_retention_mode=True,
       extra_form_fields={"use_ood_detector": True},
   )
   ```

5. **Custom transport** - keep the LiveKit integration logic but swap how audio reaches
   Detect. This is useful for a streaming Detect backend, a gateway, or tests.

   ```python
   detect = resemble.ResembleDetect(transport=my_detect_transport)
   ```

For a sensitive action such as a password reset, request a fresh check before proceeding:

```python
detect.check_now()
```

`fake_detected` is still emitted for every raw window that crosses `fake_threshold`.
Production agents should usually act on `synthetic_detected`, which applies the configured
agreement policy.

The default REST transport uploads short WAV windows directly to Detect with `Prefer: wait`
and `zero_retention_mode=True`. Pass `extra_form_fields` for advanced Detect options, or pass
a custom `transport` when using a streaming Detect backend or gateway.

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
