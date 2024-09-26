# Trigger Phrase Initiated Agent

This example demonstrated an agent that only responds to the user queries if the user provided trigger phrase is validated in the beginning of a speech.

The trigger phrase can be edited by changing this line:

```
trigger_phrase = "Hi Bob!"
```

The example uses Deepgram's STT, OpenAI's LLM, and ElevenLabs' TTS, but can be switched to other plugins as well.

## Running the example

```bash
export LIVEKIT_URL=<your LiveKit server URL>
export LIVEKIT_API_KEY=<your API Key>
export LIVEKIT_API_SECRET=<your API Secret>
export DEEPGRAM_API_KEY=<your Deepgram API key>
export OPENAI_API_KEY=<your OpenAI API key>
export ELEVEN_API_KEY=<your ElevenLabs API key>
python3 agent.py start
```
