# Speech-to-text

This example shows how you can transcript real-time audio data into text.

It uses Deepgram's STT API to transcript the audio data. It can be switched to
other STT providers by changing this line:

```python
stt = deepgram.STT()
```

All transcriptions are sent to clients in the room with LiveKit's transcription protocol.

It's currently supported in the JS SDK and React Components. This will be made available for
all other SDKs in the coming weeks.

## Running the example

```bash
export DEEPGRAM_API_KEY=your-api-key
python3 deepgram_stt.py start
```
