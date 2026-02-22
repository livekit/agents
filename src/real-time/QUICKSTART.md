# Real-Time Vision + Avatar Agent - Quick Start Guide

Get up and running in 5 minutes.

## Prerequisites

✅ Confirm you have:
- Python 3.9+
- Node.js 18+
- Git
- API keys: OpenAI, Deepgram, ElevenLabs, Anam
- LiveKit server running locally: `livekit-server --dev`

## 1. Setup Backend

```bash
# Navigate to backend
cd src/real-time/agent-py

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Create .env.local at project root (../../.env.local)
cat > ../../.env.local << 'EOF'
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret

OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...

ANAM_API_KEY=...
ANAM_AVATAR_ID=...
EOF

# Run agent in terminal mode (no LiveKit needed)
python src/agent.py console

# Or run in dev mode (connects to LiveKit)
python src/agent.py dev
```

## 2. Setup Frontend

In a new terminal:

```bash
# Navigate to frontend
cd src/real-time/frontend

# Install dependencies
npm install
# or: pnpm install

# Create .env.local
cat > .env.local << 'EOF'
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
EOF

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 3. Test the Connection

1. **Backend running?** Should see agent logs in terminal
2. **Frontend loading?** Should see "Connecting to agent..." in browser
3. **Audio working?** Try the `console` mode first for audio I/O testing

## 4. Next Steps

### Customize Agent

Edit `agent-py/src/agent.py`:

```python
class RealtimeAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your custom instructions here..."
        )
```

### Customize UI

Edit `frontend/lib/app-config.ts`:

```typescript
const defaultConfig: AppConfig = {
  enableAvatar: true,
  enableVision: true,
  enableForm: false,  // Set true to enable form
  enableChat: true,
  // ... more options
};
```

### Add Form Fields

Edit `frontend/components/app/app-view.tsx` and add FormPanel implementation.

## Troubleshooting

### "Connection refused"
→ Start LiveKit: `livekit-server --dev`

### "ANAM_API_KEY is not set"
→ Add keys to `.env.local` and restart backend

### "No video appearing"
→ Grant camera permissions in browser

### Backend crashes on startup
→ Check Python version: `python3 --version` (need 3.9+)
→ Reinstall: `pip install -e ".[dev]"`

## File Structure

```
src/real-time/
├── agent-py/              # Python backend
│   ├── src/agent.py
│   └── pyproject.toml
├── frontend/              # Next.js frontend
│   ├── app/
│   ├── components/
│   └── package.json
├── README.md              # Full documentation
└── DEPLOYMENT.md          # Production guide
```

## Key Documentation

- [Backend README](./agent-py/README.md) - Detailed backend setup
- [Frontend README](./frontend/README.md) - Frontend customization
- [Main README](./README.md) - Architecture overview
- [Deployment Guide](./DEPLOYMENT.md) - Production deployment

## Common Customizations

### Change TTS Voice

In `agent.py`, find ElevenLabs voice ID and update:

```python
tts=inference.TTS(
    model="elevenlabs/eleven_turbo_v2_5",
    voice="new-voice-id",  # Change this
    sample_rate=16000,
),
```

Get voice IDs from [ElevenLabs API](https://api.elevenlabs.io/docs).

### Change LLM Model

```python
llm=inference.LLM(model="openai/gpt-4o"),  # Use GPT-4o instead
```

### Modify Agent Instructions

```python
instructions="You are a customer service agent. Be helpful and professional."
```

### Enable Form

In `frontend/lib/app-config.ts`:

```typescript
enableForm: true,
```

Then implement form handling in `components/app/app-view.tsx`.

## Performance Tips

- **Mobile**: Set `mobileVideoQuality: "medium"` for battery savings
- **Latency**: Enable `preemptive_generation` in agent for faster responses
- **Bandwidth**: Reduce video frame sampling (edit `app-view.tsx`)

## Next Commands

```bash
# Backend development
python src/agent.py dev        # With LiveKit
python src/agent.py console    # Terminal only
python src/agent.py start      # Production

# Frontend development
npm run build                  # Production build
npm run lint                   # Code quality
tsc --noEmit                   # Type check
```

## Support

- **LiveKit Issues**: [docs.livekit.io](https://docs.livekit.io)
- **OpenAI Issues**: [platform.openai.com/docs](https://platform.openai.com/docs)
- **Next.js Issues**: [github.com/vercel/next.js](https://github.com/vercel/next.js)

## What's Next?

1. ✅ Get backend and frontend running
2. ✅ Test with terminal/dev modes
3. ⏭️ Customize instructions and UI
4. ⏭️ Add form fields and logic
5. ⏭️ Deploy to production (see DEPLOYMENT.md)

Happy building! 🚀
