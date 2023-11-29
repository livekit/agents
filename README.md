<!--BEGIN_BANNER_IMAGE-->
<!--END_BANNER_IMAGE-->

# LiveKit Agent Framework

The Agent Framework is designed for building real-time, programmable participants
that run on servers. Easily tap into LiveKit WebRTC sessions and process or generate
audio, video, and data streams.

The framework includes plugins for common workflows, such as voice activity detection and speech-to-text.

Furthermore, it integrates seamlessly with LiveKit server, offloading job queuing and scheduling responsibilities to it. This approach eliminates the need for additional queuing infrastructure. The code developed on your local machine is fully scalable when deployed to a server, supporting thousands of concurrent sessions.

## Getting Started

To install the core agent library:

```bash
pip install livekit-agents
```

Plugins can be installed individually depending on what your agent needs. Available plugins:

- livekit-plugins-elevenlabs
- livekit-plugins-google
- livekit-plugins-openai
- livekit-plugins-vad

## Terminology

- **Agent**: A function that defines the workflow of the server-side participant. This is what you will be developing.
- **Worker**: A container process responsible for managing job queuing with LiveKit server. Each worker is capable of running multiple agents simultaneously.
- **Plugin**: A library class that perform a specific task like speech-to-text with a specific provider. Agents can combine multiple plugins together to perform more complex tasks.

## Creating an Agent

Let's begin with a simple agent that performs speech-to-text on incoming audio tracks.

```python

```

The following agent listens to audio tracks and
sends a DataChannel into the room whenever speaking has been detected.

```python
import asyncio
from typing import AsyncIterator

from livekit.rtc as rtc
from livekit.agents import Agent, Processor
from livekit.processors.vad import VAD


class MyAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vad = VAD(silence_threshold_ms=250)

    def on_audio_track(self, track: rtc.Track, participant: rtc.Participant):
        loop = asyncio.get_event_loop()
        loop.create_task(self.process(track))

    async def process(self, participant: rtc.Participant, audio_track: rtc.Track):
        audio_stream = rtc.AudioStream(audio_track)
        vad_processor = Processor(process=self.vad.push_frame)
        asyncio.create_task(self.vad_result_loop(vad_processor.stream()))
        async for frame in audio_stream:
            vad_processor.push(frame)

    async def vad_result_loop(self, participant: rtc.Participant, queue: AsyncIterator[VAD.Event]):
        async for event in queue:
            if event.type == "voice_started":
                await self.participant.publish_data(f"{participant.identity} started talking")
            elif event.type == "voice_finished":
                await self.participant.publish_data(f"{participant.identity} stopped talking")

    def should_process(self, track: rtc.TrackPublication, participant: rtc.Participant) -> bool:
        return track.kind == rtc.TrackKind.Audio
```

## More Examples

Examples can be found in the `examples/` repo.

Examples coming soon:

- Siri-like
- Audio-to-Audio Language Translation
- Transcription
- Face-Detection
- Voice-to-Image

<!--BEGIN_REPO_NAV-->
<!--END_REPO_NAV-->
