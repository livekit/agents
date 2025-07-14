# Speech-to-text

This example shows realtime transcription from voice to text.

It uses OpenAI's Whisper STT API, but supports other STT plugins by changing this line:

```python
stt = openai.STT()
```

To render the transcriptions into your client application, refer to the [full documentation](https://docs.livekit.io/agents/voice-agent/transcriptions/).

## Running the example

```bash
export LIVEKIT_URL=wss://yourhost.livekit.cloud
export LIVEKIT_API_KEY=livekit-api-key
export LIVEKIT_API_SECRET=your-api-secret
export OPENAI_API_KEY=your-api-key

python3 transcriber.py start
```

Then connect to any room. For an example frontend, you can use LiveKit's [Agents Playground](https://agents-playground.livekit.io/).
