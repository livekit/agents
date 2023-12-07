# Python Agents Examples

This directory contains a full app with different examples of Agents to interact with.

# Run An Example

## Setup LiveKit

First, create a [LiveKit Cloud](https://cloud.livekit.io). (There's a free tier that does not require payment information)

Alternatively - you can run LiveKit Open Source.

Fill an `.env` file in the directory of the agent you wish to run with the keys and secrets you need. See `.env.example` for a template.

## Install Dependencies

> **_OPTIONAL:_** It's common practice to setup a virtual environment when running python projects so its dependencies don't conflict with other projects or global dependencies you've installed. To do this run `python -m venv venv` then `source venv/bin/activate`

cd into the agent you want to run (located in `agents/`) then run:

`pip install -r requirements.txt`

## Run Agents Service

LiveKit orchestrates adding agents in response to events that happen (room creation for example).

When a Worker is created, it registers itself with LiveKit and LiveKit will begin to schedule rooms
