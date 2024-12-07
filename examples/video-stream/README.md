# Video and Audio Synchronization Examples

This example demonstrates how to synchronize video and audio streams using the `AVSynchronizer` utility.

## AVSynchronizer Usage

The `AVSynchronizer` helps maintain synchronization between video and audio frames. The key principle is to push the initial synchronized video and audio frames together. After that, subsequent frames will be automatically synchronized according to the configured video FPS and audio sample rate.

```python
av_sync = AVSynchronizer(
    audio_source=audio_source,
    video_source=video_source,
    video_fps=30.0,
    video_queue_size_ms=100
)

# Push frames to synchronizer
await av_sync.push(video_frame)
await av_sync.push(audio_frame)
```

## Examples

### 1. Video File Playback (`video_play.py`)
Shows how to stream video and audio from separate sources while maintaining sync:

- Reads video and audio streams separately from a media file
- Uses separate tasks to push video and audio frames to the synchronizer
- Since the streams are continuous, a larger `queue_size_ms` can be used, though this will increase memory usage

### 2. Audio Visualization (`audio_wave.py`) 
Demonstrates generating video based on audio input:

- Generates audio frames with alternating sine waves and silence
- Creates video frames visualizing the audio waveform
- Shows how to handle cases with and without audio:
  - When audio is present: Push synchronized video and audio frames
  - During silence: Push only video frames
- Since video and audio frames are pushed in the same loop, audio frames must be smaller than the audio source queue size to avoid blocking
- Uses a small `queue_size_ms` (e.g. 50ms) to control frame generation speed during silence periods
