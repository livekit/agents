# python-agents

As we've introduced more client SDKs meant to be run in a server-side environment, we've noticed common patterns and use cases emerge. To tackle these common patterns and to reduce boilerplate, we've introduced Agents. 

Agents are LiveKit participants that can be used without having to manually wire up things like selective track subscription, track subscribed callbacks, etc.

This repo also contains common processors that are useful for agent developers. For example, many agents will need VAD (Voice Activity Detection) so we've made a processor for that.

## Getting Started

To install the core agent library:

```bash
pip install livekit-agents
```

Processors can be installed one-by-one depending on what your agent needs:

```bash
pip install livekit-processors-vad livekit-processors-openai
```

## Creating An Agent

An agent is a class that sub-classes `Agent`.

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
- Transcribtion
- Face-Detection
- Voice-to-Image