# LiveKit Plugins Google

Agent Framework plugin for services from Google Cloud. Currently supporting Google's [Speech-to-Text](https://cloud.google.com/speech-to-text) API.

## Installation

```bash
pip install livekit-plugins-google
```

## Pre-requisites

For credentials, you'll need a Google Cloud account and obtain the correct credentials. Credentials can be passed directly or via Application Default Credentials as specified in [How Application Default Credentials works](https://cloud.google.com/docs/authentication/application-default-credentials).

To use the STT and TTS API, you'll need to enable the respective services for your Google Cloud project.

- Cloud Speech-to-Text API
- Cloud Text-to-Speech API


## Gemini Multimodal Live

Gemini Multimodal Live can be used with the `MultimodalAgent` class. See examples/multimodal_agent/gemini_agent.py for an example.

### Live Video Input (experimental)

You can push video frames to your Gemini Multimodal Live session alongside the audio automatically handled by the `MultimodalAgent`.  The basic approach is to subscribe to the video track, create a video stream, sample frames at a suitable frame rate, and push them into the RealtimeSession:

```
# Make sure you subscribe to audio and video tracks
await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

# Create your RealtimeModel and store a reference
model = google.beta.realtime.RealtimeModel(
    # ...
)

# Create your MultimodalAgent as usual
agent = MultimodalAgent(
    model=model,
    # ...
)

# Async method to process the video track and push frames to Gemini
async def _process_video_track(self, track: Track):
    video_stream = VideoStream(track)
    last_frame_time = 0
    
    async for event in video_stream:
        current_time = asyncio.get_event_loop().time()
        
        # Sample at 1 FPS
        if current_time - last_frame_time < 1.0: 
            continue
            
        last_frame_time = current_time
        frame = event.frame
        
        # Push the frame into the RealtimeSession
        model.sessions[0].push_video_frame(frame)
        
    await video_stream.aclose()

# Subscribe to new tracks and process them
@ctx.room.on("track_subscribed")
def _on_track_subscribed(track: Track, pub, participant):
    if track.kind == TrackKind.KIND_VIDEO:
        asyncio.create_task(self._process_video_track(track))
```



