# Python Agents Examples

This directory contains a full app with different examples of Agents to interact with.

# Run The Examples

## Setup LiveKit

First, create a [LiveKit Cloud](https://cloud.livekit.io). (There's a free tier that does not require payment information)

Alternatively - you can run LiveKit Open Source.

Fill the `.env` file with the keys and secrets you need for the examples you wish to run. See `.env.example` for a template.

## Install Dependencies

> **_OPTIONAL:_**  It's common practice to setup a virtual environment when running python projects so its dependencies don't conflict with other projects or global dependencies you've installed. To do this run `python -m venv venv` then `source venv/bin/activate`

`pip install -r requirements.txt`

## Start the example app

To run the examples simply call

```bash
python main.py
```

After calling `main.py` a website will be running where you can interact with the various agents found in `agents/`

This will:
- Spin up a web-frontend on localhost:3000
- Spin up a python backend that instantiates and runs the agents (running on localhost:8000)