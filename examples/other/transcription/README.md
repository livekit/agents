# Speech-to-text

These examples show realtime transcription from voice to text.

- [`transcriber.py`](./transcriber.py) transcribes one remote participant.
- [`multi-user-transcriber.py`](./multi-user-transcriber.py) starts one transcription session for each remote participant in the room.

`transcriber.py` uses OpenAI's Whisper STT API, but supports other STT plugins by changing this line:

```python
stt = openai.STT()
```

`multi-user-transcriber.py` uses LiveKit Inference with Deepgram:

```python
stt = inference.STT("deepgram/nova-3")
```

To render the transcriptions in your client application, refer to the [text and transcriptions documentation](https://docs.livekit.io/agents/multimodality/text/).

## Running the examples

From the repository root, install dependencies:

```bash
uv sync --all-extras --dev
```

Create an `examples/.env` file with your LiveKit credentials and the provider credentials for the example you run:

```bash
LIVEKIT_URL=wss://yourhost.livekit.cloud
LIVEKIT_API_KEY=livekit-api-key
LIVEKIT_API_SECRET=your-api-secret
OPENAI_API_KEY=your-api-key
```

For single-participant transcription:

```bash
uv run examples/other/transcription/transcriber.py dev
```

For multi-user transcription:

```bash
uv run examples/other/transcription/multi-user-transcriber.py dev
```

Then connect one or more participants to a room and dispatch the agent to that same room. For an example frontend, you can use LiveKit's [Agents Playground](https://agents-playground.livekit.io/).

When `multi-user-transcriber.py` is working, the agent logs a line similar to this for each participant that speaks:

```text
participant-identity -> hello from this participant
```

The example also publishes transcripts to the room with text output enabled, so clients can render them from the `lk.transcription` text stream topic.

## Troubleshooting

If the agent starts but does not transcribe:

- Confirm the frontend participant and the agent are connected to the same room.
- Confirm at least one remote participant is publishing microphone audio.
- Confirm the participant is not the agent itself. The multi-user example transcribes remote participants only.
- Confirm `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` are loaded from `examples/.env` or your shell.
- For `transcriber.py`, confirm `OPENAI_API_KEY` is set. For `multi-user-transcriber.py`, confirm your LiveKit project can use LiveKit Inference.
