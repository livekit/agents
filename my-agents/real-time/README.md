# Anam healthcare intake assistant

A LiveKit Agents demo that guides patients through a medical intake form using voice and an Anam lip-synced avatar. The agent (Liv) collects information one field at a time and updates the form in real time via RPC to the frontend. The demo includes both a **Python** and a **TypeScript** backend; run whichever you prefer. The frontend works with either.

## Features

- **Voice intake flow**: Liv asks for each form field in order (name, DOB, address, phone, emergency contact, medications, allergies, reason for visit)
- **One question at a time**: Confirms each answer before moving to the next field
- **Real-time form updates**: Agent updates the on-screen form as the user speaks via RPC
- **Anam avatar**: Lip-synced video avatar powered by the Anam plugin
- **Voice pipeline**: Deepgram STT, OpenAI LLM, and ElevenLabs TTS (via LiveKit Inference)
- **Form submission**: Final confirmation loop, then submit from the frontend

## Prerequisites

- **For the Python backend**: Python 3.10+, uv
- **For the TypeScript backend**: Node.js, pnpm
- **For the frontend**: Node.js, pnpm
- LiveKit account
- Anam API key

## Installation

1. Clone this repository and go to the anam demo
   ```bash
   git clone <repository-url>
   cd complex-agents/avatars/anam
   ```

2. Install the backend you want to use (or both).
   - **Python backend**
     ```bash
     cd agent-py
     uv sync
     cd ..
     ```
   - **TypeScript backend**
     ```bash
     cd agent-ts
     pnpm install
     cd ..
     ```

3. Install frontend dependencies
   ```bash
   cd frontend
   pnpm install
   cd ..
   ```

4. Set up environment for your chosen agent
   - **Python**: In `agent-py`, copy `.env.example` to `.env.local` and add your LiveKit and Anam credentials.
   - **TypeScript**: In `agent-ts`, copy `.env.example` to `.env.local` and add your LiveKit and Anam credentials.

5. Set up environment for the frontend  
   In `frontend`, copy `.env.example` to `.env.local`. Add the same LiveKit credentials and set `AGENT_NAME=Anam-Demo` so the frontend dispatches to the agent.

## Configuration

Enter your environment variables in the `.env.local` file:

```bash
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
ANAM_API_KEY=your_anam_api_key
```

Get your Anam API key from the [Anam dashboard](https://lab.anam.ai/api-keys). You can get your LiveKit environment variables from the [LiveKit dashboard](https://cloud.livekit.io/) or load them from the [LiveKit CLI](https://docs.livekit.io/home/cli/cli-setup):

```bash
lk cloud auth
lk app env -w -d .env.local
```

Run that from either `agent-py` or `agent-ts` depending on which backend you use.

### Frontend (`frontend/.env.local`)

```bash
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
LIVEKIT_URL=wss://your-project.livekit.cloud
AGENT_NAME=Anam-Demo
```

Use the same LiveKit values as the agent. `AGENT_NAME=Anam-Demo` matches both backends so the frontend connects to whichever agent you run.

## Usage

Run **one** backend (Python or TypeScript) and the frontend. Do not run both backends at the same time.

### Python backend

1. One-time: download Silero VAD and the turn detector model
   ```bash
   cd agent-py
   uv run python src/agent.py download-files
   ```

2. Start the agent
   ```bash
   uv run python src/agent.py dev
   ```

The agent registers with your LiveKit project as `Anam-Demo`.

### TypeScript backend

1. One-time: download Silero VAD and the turn detector model
   ```bash
   cd agent-ts
   pnpm run download-files
   ```

2. Start the agent
   ```bash
   pnpm run dev
   ```

The agent registers with your LiveKit project as `Anam-Demo`.

### Frontend

1. Open a new terminal and go to the frontend:
   ```bash
   cd frontend
   ```

2. Start the dev server:
   ```bash
   pnpm dev
   ```

3. In your browser, open:
   ```
   http://localhost:3000
   ```

4. Start a session and complete the intake form with Liv.

## Project structure

```
anam/
├── agent-py/              # Python backend (optional)
│   ├── src/agent.py       # Intake agent and Anam avatar session
│   ├── pyproject.toml
│   └── .env.local
├── agent-ts/              # TypeScript backend (optional)
│   ├── src/               # Agent entry and intake logic
│   ├── package.json
│   └── .env.local
└── frontend/
    ├── app/               # Next.js app router
    ├── components/app/    # Session, avatar, intake form UI
    └── .env.local         # LiveKit credentials + AGENT_NAME
```

Both agents use the same agent name (`Anam-Demo`) and the same function tools (`update_field`, `get_form_state`, `submit_form`) to call RPCs on the frontend so the form stays in sync with the conversation.

## Built with

- [LiveKit Agents](https://docs.livekit.io/agents/) - Agent framework
- [Anam plugin](https://docs.livekit.io/agents/models/avatar/plugins/anam/) - Lip-synced avatar
- [LiveKit Inference](https://docs.livekit.io/agents/models/) - STT, LLM, TTS models
