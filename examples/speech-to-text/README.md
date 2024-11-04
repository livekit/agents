# Speech-to-text

This example show realtime transcription from audio to text.

It uses Deepgram's STT API, but supports other STT plugins by changing this line:

```python
stt = deepgram.STT()
```

To render the transcriptions into your client application, refer to the [full documentation](https://docs.livekit.io/agents/voice-agent/transcriptions/).

## Running the example

```bash
export DEEPGRAM_API_KEY=your-api-key
python3 deepgram_stt.py start
```
