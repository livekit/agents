# Participant Entrypoint Example

This example shows how to do things when participants join. For example, a common use case is to fetch some external data based on the participant's attributes.

## Run

### Setup and activate a virtual env:

`python -m venv venv`

`source venv/bin/activate`

### Set environment variables:

```bash
export LIVEKIT_URL=<your LiveKit server URL>
export LIVEKIT_API_KEY=<your API Key>
export LIVEKIT_API_SECRET=<your API Secret>
```

### Install requirments:
`pip install -r requirements.txt`

### Run the agent worker:

`python participant_entrypoint.py dev`

### Test with a LiveKit frontend:

We've built [Agents Playground](https://agents-playground.livekit.io) so you don't have to build your own frontend while you iterate on your agent.
