# Trillet Voiceguard Plugin

A real-time AI voice detection plugin for LiveKit that identifies synthetic/fake voices during voice calls. This plugin analyzes audio streams during the first 30 seconds of calls and can automatically terminate calls when fake voices are detected.

## Features

- 🎯 **Real-time Voice Detection**: Analyzes audio streams in real-time using AI voice detection
- 🔊 **Audio Processing**: Converts audio to optimal format (16-bit PCM, Mono, 16kHz)
- ⚡ **WebSocket Integration**: Streams audio data to Trillet Voiceguard service via WebSocket
- 🛡️ **Automatic Call Protection**: Optionally terminates calls when fake voices are detected
- 📊 **Comprehensive Reporting**: Provides detailed analysis results and summaries
- 🎚️ **Configurable Parameters**: Customizable detection parameters and thresholds
- 💾 **Audio Recording**: Optional audio recording for archival purposes

## Installation

```bash
pip install livekit-plugins-trillet-antibot
```

## Authentication Setup

Before using the plugin, you'll need to obtain your authentication credentials:

1. **Visit [app.trillet.ai](https://app.trillet.ai)** and create an account or sign in
2. **Navigate to your dashboard** to find your account credentials
3. **Copy your API key** - This authenticates your requests to the Voiceguard service
4. **Copy your workspace ID** - This identifies your workspace for the detection service

Keep these credentials secure and never commit them to version control.

## Quick Start

```python
from livekit.plugins.trillet_antibot import TrilletAntiBot
from livekit import api, rtc
import asyncio

async def voice_detection_example():
    # Connect to LiveKit room
    room = rtc.Room()
    
    # Create antibot instance 
    antibot = TrilletAntiBot(
        room=room,
        ctx="participant_context",
        ai_key="your-voiceguard-api-key",
        workspace_id="your-workspace-id"
    )
    
    # Start voice detection
    await antibot.start_streaming()

# Run the example
asyncio.run(voice_detection_example())
```

## Configuration Parameters

### Required Parameters

- **`room`**: LiveKit Room object - The active LiveKit room to monitor
- **`ctx`**: Context object - LiveKit context containing API client
- **`ai_key`**: str - Your Trillet Voiceguard API key
- **`workspace_id`**: str - Your Trillet Voiceguard workspace ID

> **📝 Note**: You can obtain your API key and workspace ID by visiting [app.trillet.ai](https://app.trillet.ai) and accessing your account dashboard.

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `duration_seconds` | int | 30 | Duration to run voice detection (seconds) |
| `segment_length` | float | 3.0 | Length of each audio segment for analysis (seconds) |
| `confidence_threshold` | float | 0.70 | Confidence threshold for fake voice detection (0.0-1.0) |
| `terminate_on_fake` | bool | False | Whether to automatically terminate calls when fake voice is detected |
| `save_audio_to_s3` | bool | False | Whether to record and save audio for archival |

### Parameter Details

#### `duration_seconds` (int, default: 30)
Controls how long the voice detection runs. The service will automatically stop after this duration.
```python
# Run detection for 60 seconds
antibot = TrilletAntiBot(room, ctx, ai_key, workspace_id, duration_seconds=60)
```

#### `segment_length` (float, default: 3.0)
Defines the length of each audio segment sent for analysis. Shorter segments provide more frequent results but may be less accurate.
```python
# Use 5-second segments for analysis
antibot = TrilletAntiBot(room, ctx, ai_key, workspace_id, segment_length=5.0)
```

#### `confidence_threshold` (float, default: 0.70)
Sets the minimum confidence level required to classify a voice as fake. Higher values reduce false positives but may miss some fake voices.
```python
# Use higher threshold for more conservative detection
antibot = TrilletAntiBot(room, ctx, ai_key, workspace_id, confidence_threshold=0.85)
```

#### `terminate_on_fake` (bool, default: False)
Controls whether calls should be automatically terminated when fake voices are detected.
```python
# Enable automatic call termination
antibot = TrilletAntiBot(room, ctx, ai_key, workspace_id, terminate_on_fake=True)
```

#### `save_audio_to_s3` (bool, default: False)
Enables recording of audio during detection for archival purposes.
```python
# Enable audio recording
antibot = TrilletAntiBot(room, ctx, ai_key, workspace_id, save_audio_to_s3=True)
```

## Usage Examples

### Basic Usage with Default Settings
```python
antibot = TrilletAntiBot(room, ctx, "your_api_key", "your_workspace_id")
await antibot.start_streaming()
```

### Voice Detection with Audio Recording
```python
antibot = TrilletAntiBot(
    room=room,
    ctx=ctx,
    ai_key="your_api_key",
    workspace_id="your_workspace_id",
    save_audio_to_s3=True
)
await antibot.start_streaming()

# Check if audio was saved after detection
upload_status = antibot.get_audio_upload_status()
if upload_status and upload_status.get('success'):
    print(f"Audio archived successfully")
```

### Conservative Detection with Call Termination
```python
antibot = TrilletAntiBot(
    room=room,
    ctx=ctx,
    ai_key="your_api_key",
    workspace_id="your_workspace_id",
    confidence_threshold=0.85,
    terminate_on_fake=True,
    save_audio_to_s3=True
)
await antibot.start_streaming()
```

### Extended Monitoring Without Termination
```python
antibot = TrilletAntiBot(
    room=room,
    ctx=ctx,
    ai_key="your_api_key",
    workspace_id="your_workspace_id",
    duration_seconds=60,
    terminate_on_fake=False
)
await antibot.start_streaming()

# Check results after completion
if antibot.is_voice_likely_fake():
    print("Fake voice detected!")
    summary = antibot.get_final_summary()
    print(f"Fake percentage: {summary.get('fake_percentage', 0)}%")
```

### Custom Segment Analysis
```python
antibot = TrilletAntiBot(
    room=room,
    ctx=ctx,
    ai_key="your_api_key",
    workspace_id="your_workspace_id",
    segment_length=2.0,  # Analyze every 2 seconds
    duration_seconds=45
)
await antibot.start_streaming()
```

## API Methods

### Control Methods

#### `start_streaming()`
Starts the voice detection process. Returns immediately while detection runs in the background.
```python
await antibot.start_streaming()
```

#### `stop_streaming()`
Stops the voice detection process and cleans up resources.
```python
await antibot.stop_streaming()
```

### Data Retrieval Methods

#### `get_results() -> List[Dict[str, Any]]`
Returns all individual detection results received so far.
```python
results = antibot.get_results()
for result in results:
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence_fake']}")
```

#### `get_final_summary() -> Optional[Dict[str, Any]]`
Returns the final summary from the Voiceguard service (available after detection completes).
```python
summary = antibot.get_final_summary()
if summary:
    print(f"Overall prediction: {summary['overall_prediction']}")
    print(f"Total segments: {summary['total_segments']}")
    print(f"Fake percentage: {summary['fake_percentage']}%")
```

#### `get_call_termination_reason() -> Optional[str]`
Returns the reason for call termination if the call was terminated due to fake voice detection.
```python
reason = antibot.get_call_termination_reason()
if reason == "fake_voice_detected":
    print("Call was terminated due to fake voice detection")
```

#### `get_audio_upload_status() -> Optional[Dict[str, Any]]`
Returns the status and details of audio archival (if enabled).
```python
upload_status = antibot.get_audio_upload_status()
if upload_status:
    if upload_status.get('success'):
        print("Audio archived successfully")
        print(f"File size: {upload_status.get('file_size')} bytes")
    else:
        print(f"Archival failed: {upload_status.get('error')}")
```

#### `is_voice_likely_fake(threshold: float = 0.7) -> bool`
Determines if the voice is likely fake based on detection results.
```python
if antibot.is_voice_likely_fake(threshold=0.8):
    print("Voice appears to be fake with high confidence")
```

## Response Format

### Individual Detection Results
Each detection result contains:
```python
{
    "timestamp": 1642684800.123,
    "elapsed_time": 3.5,
    "prediction": "FAKE",  # or "REAL"
    "confidence_fake": 0.85,
    "confidence_real": 0.15
}
```

### Final Summary
The final summary includes:
```python
{
    "timestamp": 1642684830.456,
    "elapsed_time": 30.0,
    "type": "final_summary",
    "overall_prediction": "FAKE",
    "total_segments": 10,
    "fake_predictions": 7,
    "real_predictions": 3,
    "fake_percentage": 70.0,
    "average_fake_confidence": 0.78,
    "average_real_confidence": 0.22,
    # Audio archival info (if save_audio_to_s3=True)
    "audio_upload": {
        "success": True,
        "filename": "voiceguard_audio_room1_20240126_143022.wav",
        "uploaded_at": "20240126_143022",
        "file_size": 960000
    }
}
```

### Audio Upload Status
When `save_audio_to_s3=True`, the audio archival status contains:
```python
{
    "success": True,  # or False if archival failed
    "filename": "voiceguard_audio_roomname_timestamp.wav",
    "uploaded_at": "20240126_143022",
    "file_size": 960000,
    "error": "Error message if success=False"
}
```

## Error Handling

The plugin includes comprehensive error handling and logging:

```python
import logging

# Enable debug logging to see detailed information
logging.getLogger().setLevel(logging.DEBUG)

try:
    antibot = TrilletAntiBot(room, ctx, ai_key, workspace_id)
    await antibot.start_streaming()
except Exception as e:
    print(f"Error starting voice detection: {e}")
```

## Audio Requirements

The plugin automatically handles audio format conversion:
- **Input**: Any LiveKit-supported audio format
- **Output to Service**: 16-bit PCM, Mono, 16kHz
- **Participants**: Processes audio from remote participants (incoming audio)
- **Recording Format**: WAV files (16-bit PCM, Mono, 16kHz) when audio recording is enabled

## WebSocket Connection

The plugin connects to the Trillet Voiceguard service via WebSocket:
- **URL**: `wss://p01--trillet-voice-guard-dev--j629vb9mq7pk.ccvhxjx8pb.code.run/trillet-voiceguard/ws/detect`
- **Protocol**: Binary audio data in, JSON responses out
- **Auto-reconnection**: Not implemented (single-session design)

## Logging

The plugin provides detailed logging at various levels:
- **INFO**: General operation status and results
- **DEBUG**: Detailed processing information
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors and failures

## Best Practices

1. **API Key Security**: Store your API key securely and never commit it to version control
2. **Resource Cleanup**: Always call `stop_streaming()` or use proper cleanup to avoid resource leaks
3. **Error Handling**: Implement proper error handling for network issues and API failures
4. **Threshold Tuning**: Test different confidence thresholds to find the optimal balance for your use case
5. **Monitoring**: Monitor the logs for connection issues and detection accuracy
6. **Audio Privacy**: Ensure compliance with privacy regulations when recording and storing audio data

## Troubleshooting

### Common Issues

#### WebSocket Connection Failures
```
ERROR: Failed to connect to Voiceguard websocket
```
- Check your API key is valid
- Verify network connectivity
- Ensure the service is available

#### No Audio Data
```
WARNING: No audio data being processed
```
- Ensure participants are speaking
- Check that audio tracks are being received
- Verify room has remote participants

#### High False Positives
- Increase `confidence_threshold` parameter
- Consider using longer `segment_length` for more stable results

#### Detection Not Running
- Ensure `start_streaming()` was called
- Check logs for connection errors
- Verify `duration_seconds` hasn't expired

#### S3 Upload Issues
```
ERROR: Error uploading audio to S3
```
- Check your API key has upload permissions
- Verify network connectivity
- Check server logs for detailed error information

#### Audio Recording Not Working
- Ensure `save_audio_to_s3=True` is set
- Check that audio data is being received from participants
- Verify the room has active audio streams

## Requirements

- Python 3.8+
- LiveKit Python SDK
- WebSocket support (websockets >= 11.0)
- NumPy (for audio processing)
- aiohttp (for S3 upload functionality)
- Wave support (built-in Python module)

## License

[Your license information here]

## Support

For support and questions:
- Email: [Your support email]
- Documentation: [Your documentation URL]
- Issues: [Your issue tracker URL]
