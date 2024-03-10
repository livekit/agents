# KITT - A Talking AI Example Using Deepgram, ChatGPT, and ElevenLabs

## Running locally

To run the KITT agent locally, install its dependencies:

```bash
pip install -r requirements.txt
```

then start the worker:

```bash
python kitt.py start --api-key=<your livekit api key> --api-secret=<your livekit api secret> --url=<your livekit ws url>
```

This starts the worker and will be listening for new job requests. This worker is configured to listen to the "room" job type (`JT_ROOM`) which means it will get a JobRequest when a new LiveKit room is created.

## How it works

KITT has 3 stages:

- Speech-to-text (STT)
- LLM
- Text-to-speech (TTS)

Speech-to-text uses `livekit-plugins-deepgram` and text-to-speech uses `livekit-plugins-elevenlabs`.

When a KITT agent starts, it publishes an audio track right away and sends an intro message. 

It then subscribes to any existing and new audio tracks and sends their rtc.AudioFrames into the STT stream. 

The STT stream produces transcription results from the audio frames and when the text is final it gets sent into the LLM (ChatGPT). Results from the LLM are pushed into the TTS stream which yields rtc.AudioFrames for the agent's voice. These frames are then published into the LiveKit room.

## How to deploy

See the `Dockerfile` in the `agents/` dir for reference.
