# Demo Assistant

This is the demo voice agent that we run at kitt.livekit.io.

Uses:
- Deepgram for Speech-to-text
- OpenAI for LLM
- Elevenlabs for Text-to-speech

## Run

### Setup and activate a virtual env:

`python -m venv venv`

`source venv/bin/activate`

### Set environment variables:

```bash
export LIVEKIT_URL=<your LiveKit server URL>
export LIVEKIT_API_KEY=<your API Key>
export LIVEKIT_API_SECRET=<your API Secret>
export ELEVEN_API_KEY=<your ElevenLabs API key>
export DEEPGRAM_API_KEY=<your Deepgram API key>
export OPENAI_API_KEY=<your OpenAI API key>
```

### Install requirments:
`pip install -r requirements.txt`

### Download files (in this case, it downloads the model weights for Voice-activity-detection):

`python agent.py download-files`

### Run the agent worker:

`python agent.py dev`

### Test with a LiveKit frontend:

We've built [Agents Playground](https://agents-playground.livekit.io) so you don't have to build your own frontend while you iterate on your agent.
