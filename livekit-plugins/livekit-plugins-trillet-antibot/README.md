# Trillet Antibot Plugin

A real-time AI voice detection plugin for LiveKit that identifies synthetic/fake voices during voice calls. This plugin analyzes audio streams during the calls and can automatically terminate calls when fake voices are detected.

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
| `save_audio_to_s3` | bool | False | Opt-in to help improve our AI model by contributing audio data |

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
Opt-in to help improve our AI model by allowing us to collect and analyze your audio recordings. This helps us enhance detection accuracy for future versions.
```python
# Opt-in to help improve the model with your audio data
antibot = TrilletAntiBot(room, ctx, ai_key, workspace_id, save_audio_to_s3=True)
```

> **🤝 Help Us Improve**: By enabling this option, you're contributing to the advancement of voice detection technology. Your audio data helps us train better models and improve accuracy for everyone.

## Usage Examples

### Basic Usage with Default Settings
```python
antibot = TrilletAntiBot(room, ctx, "your_api_key", "your_workspace_id")
await antibot.start_streaming()
```

### Voice Detection with Model Improvement Contribution
```python
antibot = TrilletAntiBot(
    room=room,
    ctx=ctx,
    ai_key="your_api_key",
    workspace_id="your_workspace_id",
    save_audio_to_s3=True  # Help improve the AI model
)
await antibot.start_streaming()

# Check if your contribution was successfully submitted
upload_status = antibot.get_audio_upload_status()
if upload_status and upload_status.get('success'):
    print("Thank you for contributing to model improvement!")
```

### Conservative Detection with Model Contribution
```python
antibot = TrilletAntiBot(
    room=room,
    ctx=ctx,
    ai_key="your_api_key",
    workspace_id="your_workspace_id",
    confidence_threshold=0.85,
    terminate_on_fake=True,
    save_audio_to_s3=True  # Contribute to model training
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
Returns the status of your audio data contribution (if model improvement is enabled).
```python
upload_status = antibot.get_audio_upload_status()
if upload_status:
    if upload_status.get('success'):
        print("Audio data contributed successfully - thank you for helping improve our model!")
        print(f"File size: {upload_status.get('file_size')} bytes")
    else:
        print(f"Contribution failed: {upload_status.get('error')}")
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
}
```

> **🔒 Privacy Note**: Audio data contributed for model improvement is handled securely and used solely for enhancing our AI detection capabilities.

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
- **Model Contribution**: When enabled, records WAV files (16-bit PCM, Mono, 16kHz) to help improve AI accuracy

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
5. **Model Improvement**: Consider enabling `save_audio_to_s3=True` to contribute to model enhancement
6. **Privacy Compliance**: When contributing audio data, ensure you have appropriate consent and comply with privacy regulations

## Support

For support and questions:
- Email: support@trillet.ai
