# Participant Entrypoint Example

This example shows how to use [mem0](https://mem0.ai/)'s Enhanced memory functionality with LiveKit. This example is a simple travel guide voice assistant that uses mem0 to remember conversations and user's preferences.

## Run

### Setup and activate a virtual env:

`python -m venv venv`

`source venv/bin/activate`

### Set environment variables:
You can set the environment variables in your terminal or create a `.env` file in the root of the project like shown in `.env.example`.
```bash
export LIVEKIT_URL=<your LiveKit server URL>
export LIVEKIT_API_KEY=<your API Key>
export LIVEKIT_API_SECRET=<your API Secret>
export DEEPGRAM_API_KEY=your_deepgram_api_key
export MEM0_API_KEY=MEM0_API_KEY=your_mem0_api_key
export OPENAI_API_KEY=your_openai_api_key
```

### Install requirments:
`pip install -r requirements.txt`

### Define the User ID:
You can define the user ID in the `mem0_assistant.py` file like shown below:
```
# Define a global user ID for simplicity
USER_ID = "voice_user"
```

### Run the agent worker:

`python participant_entrypoint.py start`

### Test with a LiveKit frontend:

Connect to the agent using the LiveKit frontend [Agents Playground](https://agents-playground.livekit.io).