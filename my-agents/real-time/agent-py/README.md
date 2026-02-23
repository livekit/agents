<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# Anam Healthcare Intake Demo - Python

A healthcare intake assistant built with [LiveKit Agents for Python](https://github.com/livekit/agents), [Anam avatars](https://docs.livekit.io/agents/models/avatar/plugins/anam/), and [LiveKit Cloud](https://cloud.livekit.io/).

This demo includes:

- A healthcare intake assistant named Liv that guides patients through a medical intake form
- [Anam avatar integration](https://docs.livekit.io/agents/models/avatar/plugins/anam/) with lip-synced video avatar
- Voice AI pipeline using Deepgram (STT), OpenAI (LLM), and ElevenLabs (TTS) served through [LiveKit Inference](https://docs.livekit.io/agents/models/)
- RPC-based communication between agent and frontend for real-time form updates
- Function tools for updating form fields, retrieving form state, and submitting forms
- [LiveKit Turn Detector](https://docs.livekit.io/agents/build/turns/turn-detector/) for contextually-aware speaker detection with multilingual support
- [Background voice cancellation](https://docs.livekit.io/home/cloud/noise-cancellation/)
- Integrated [metrics and logging](https://docs.livekit.io/agents/build/metrics/)
- A Dockerfile ready for [production deployment](https://docs.livekit.io/agents/ops/deployment/)

## Coding agents and MCP

This project is designed to work with coding agents like [Cursor](https://www.cursor.com/) and [Claude Code](https://www.anthropic.com/claude-code). 

To get the most out of these tools, install the [LiveKit Docs MCP server](https://docs.livekit.io/mcp).

For Cursor, use this link:

[![Install MCP Server](https://cursor.com/deeplink/mcp-install-light.svg)](https://cursor.com/en-US/install-mcp?name=livekit-docs&config=eyJ1cmwiOiJodHRwczovL2RvY3MubGl2ZWtpdC5pby9tY3AifQ%3D%3D)

For Claude Code, run this command:

```
claude mcp add --transport http livekit-docs https://docs.livekit.io/mcp
```

For Codex CLI, use this command to install the server:
```
codex mcp add --url https://docs.livekit.io/mcp livekit-docs
```

For Gemini CLI, use this command to install the server:
```
gemini mcp add --transport http livekit-docs https://docs.livekit.io/mcp
```

The project includes a complete [AGENTS.md](AGENTS.md) file for these assistants. You can modify this file  your needs. To learn more about this file, see [https://agents.md](https://agents.md).

## Dev Setup

Clone the repository and install dependencies to a virtual environment:

```console
cd agent-starter-python
uv sync
```

Sign up for [LiveKit Cloud](https://cloud.livekit.io/) and [Anam](https://www.anam.ai/) then set up the environment by copying `.env.example` to `.env.local` and filling in the required keys:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `ANAM_API_KEY` - Get your API key from the [Anam dashboard](https://dashboard.anam.ai/)

You can load the LiveKit environment automatically using the [LiveKit CLI](https://docs.livekit.io/home/cli/cli-setup):

```bash
lk cloud auth
lk app env -w -d .env.local
```

## Run the agent

Before your first run, you must download certain models such as [Silero VAD](https://docs.livekit.io/agents/build/turns/vad/) and the [LiveKit turn detector](https://docs.livekit.io/agents/build/turns/turn-detector/):

```console
uv run python src/agent.py download-files
```

Next, run this command to speak to your agent directly in your terminal:

```console
uv run python src/agent.py console
```

To run the agent for use with a frontend or telephony, use the `dev` command:

```console
uv run python src/agent.py dev
```

In production, use the `start` command:

```console
uv run python src/agent.py start
```

## Frontend

This demo includes a custom Next.js frontend in the `../frontend` directory. The frontend features:

- Real-time video rendering of the Anam avatar
- Interactive healthcare intake form that updates as the agent fills it in
- RPC communication between frontend and agent for form updates
- Built with Next.js, React, and Tailwind CSS

To run the frontend, see the README in the `../frontend` directory.

## How it works

The Python agent:

1. Connects to the LiveKit room and sets up the voice pipeline with STT, LLM, and TTS
2. Starts an Anam avatar session with lip-sync capabilities
3. Greets the patient and asks for their full legal name to begin intake
4. Uses function tools to communicate with the frontend via RPC:
   - `update_field` - Updates a specific form field on the patient's screen
   - `get_form_state` - Retrieves the current state of all form fields
   - `submit_form` - Submits the completed intake form
5. Walks the patient through each field of the intake form one at a time
6. Confirms each answer before moving to the next field

## Resources

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [Anam Avatar Plugin](https://docs.livekit.io/agents/models/avatar/plugins/anam/)
- [LiveKit Inference](https://docs.livekit.io/agents/models/)
- [Function Tools](https://docs.livekit.io/agents/build/tools/)
- [RPC Communication](https://docs.livekit.io/home/client/data/rpc/)
