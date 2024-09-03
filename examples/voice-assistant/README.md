# Voice Assistant Example

This example shows a few usages of the VoiceAssistant class:

- `minimal_assistant.py`: a basic conversational assistant
- `function_calling_weather.py`: a weather assistant that calls an API endpoint to retrieve the weather
- `simple_rag`: a simple RAG assistant that answers questions by querying a embeddings index

The assistants use:

- Deepgram for Speech-to-text
- OpenAI for LLM
- OpenAI for Text-to-Speech

## Run

Instructions for running the two agents are identical, the following steps will assume you are running `minimal_assistant.py`

### Setup and activate a virtual env:

`python -m venv venv`

`source venv/bin/activate`

### Set environment variables:

```bash
export LIVEKIT_URL=<your LiveKit server URL>
export LIVEKIT_API_KEY=<your API Key>
export LIVEKIT_API_SECRET=<your API Secret>
export DEEPGRAM_API_KEY=<your Deepgram API key>
export OPENAI_API_KEY=<your OpenAI API key>
```

### Install requirments:

`pip install -r requirements.txt`

### Run the agent worker:

`python minimal_assistant.py dev`

### Test with a LiveKit frontend:

We've built [Agents Playground](https://agents-playground.livekit.io) so you don't have to build your own frontend while you iterate on your agent.
