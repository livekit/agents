# Real-Time Vision + Avatar Agent

Unified real-time agent with video vision (GPT-4o-vision) and animated avatars (Anam), featuring a responsive mobile-first UI for both web and mobile devices.

## Features

- **Video Vision**: Agent can see and respond to user's video feed using GPT-4o-vision multimodal LLM
- **Animated Avatars**: Lip-synced avatar responses using Anam with ElevenLabs TTS
- **Responsive UI**: Mobile-first design that adapts to all screen sizes
- **Real-time Communication**: Voice and video streaming via LiveKit WebRTC
- **Self-Hosted**: Runs on self-hosted LiveKit server (no cloud dependency)
- **RPC Methods**: Bidirectional communication between backend agent and frontend UI

## Architecture

### Backend (`agent-py/`)
- **Agent Framework**: LiveKit Agents with configurable STT/LLM/TTS pipeline
- **Vision**: GPT-4o-vision for multimodal input from user video
- **Avatar**: Anam avatar with 16kHz lip-synced TTS
- **VAD**: Silero voice activity detection with turn detection
- **Noise Cancellation**: BVC for audio preprocessing

### Frontend (`frontend/`)
- **Framework**: Next.js 15 with React 19
- **Real-time**: LiveKit Client SDK for WebRTC
- **UI**: Responsive Tailwind CSS components with mobile optimizations
- **State**: React hooks with RPC communication layer

## Setup

### Prerequisites

- Python 3.9+ (backend)
- Node.js 18+ (frontend)
- LiveKit server running locally (WebSocket on `ws://localhost:7880`)
- API keys: OpenAI, Anam, Deepgram, ElevenLabs

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd agent-py
   ```

2. **Create Python environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set environment variables:**
   Create `.env.local` in project root:
   ```bash
   LIVEKIT_URL=ws://localhost:7880
   LIVEKIT_API_KEY=devkey
   LIVEKIT_API_SECRET=secret
   
   OPENAI_API_KEY=sk-...
   DEEPGRAM_API_KEY=...
   ELEVENLABS_API_KEY=...
   
   ANAM_API_KEY=...
   ANAM_AVATAR_ID=...
   ```

5. **Run agent:**
   ```bash
   python src/agent.py console  # Terminal mode
   python src/agent.py dev      # Development mode with hot reload
   ```

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   # or
   pnpm install
   ```

3. **Set environment variables:**
   Create `.env.local`:
   ```bash
   NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880
   LIVEKIT_API_KEY=devkey
   LIVEKIT_API_SECRET=secret
   ```

4. **Run development server:**
   ```bash
   npm run dev
   ```

5. **Open in browser:**
   Navigate to `http://localhost:3000`

## Configuration

### Backend Configuration (`agent.py`)

- **USE_OPENAI_REALTIME**: Toggle between OpenAI Realtime API (low-latency) vs modular pipeline (flexible)
- **STT Model**: Deepgram Nova-3 (multi-language)
- **LLM Model**: OpenAI GPT-4o-mini or Realtime API
- **TTS Model**: ElevenLabs Turbo v2.5 at 16kHz (Anam requirement)
- **VAD**: Silero with MultilingualModel turn detection
- **Preemptive Generation**: Enabled for responsive interactions

### Frontend Configuration (`lib/app-config.ts`)

- **enableAvatar**: Show avatar video rendering
- **enableVision**: Enable vision capabilities
- **enableForm**: Show intake form
- **enableChat**: Show chat transcript
- **mobileVideoQuality**: Auto-adjust video quality on mobile networks
- **darkMode**: Light/dark/auto theme

## Mobile Responsive Design

### Breakpoints
- **xs**: 320px (mobile)
- **sm**: 640px (tablet)
- **md**: 768px (small desktop)
- **lg**: 1024px (desktop)
- **xl**: 1280px (large desktop)

### Layout Behavior

**Mobile (< 768px):**
- Vertical stack: Avatar (full width) → Form/Chat (full width)
- Compact spacing and touch-friendly buttons (44px minimum)
- Optimized for portrait orientation

**Tablet (768px - 1024px):**
- Side-by-side layout: Avatar (left) → Form/Chat (right)
- Adaptive padding and spacing

**Desktop (> 1024px):**
- Full split layout with generous spacing
- Optimized for landscape orientation

## Development

### Project Structure

```
src/real-time/
├── agent-py/
│   ├── pyproject.toml
│   └── src/
│       └── agent.py
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   ├── layout.tsx
│   │   └── api/
│   │       └── connection-details/
│   ├── components/
│   │   ├── app/
│   │   │   ├── agent-session-provider.tsx
│   │   │   └── app-view.tsx
│   │   └── ui/
│   ├── hooks/
│   │   ├── useRpcHandlers.ts
│   │   └── useVideoTrack.ts
│   ├── lib/
│   │   ├── app-config.ts
│   │   ├── utils.ts
│   │   └── rpc-types.ts
│   └── styles/
│       └── globals.css
└── README.md
```

### Building & Deployment

**Backend:**
```bash
cd agent-py
python src/agent.py start  # Production mode
```

**Frontend:**
```bash
cd frontend
npm run build
npm start
```

## Troubleshooting

### Connection Issues
- Verify LiveKit server is running: `livekit-server --dev`
- Check `LIVEKIT_URL` matches your server (default: `ws://localhost:7880`)
- Ensure API keys are correct in `.env.local`

### Avatar Not Displaying
- Confirm `ANAM_API_KEY` and `ANAM_AVATAR_ID` are set
- Verify TTS sample rate is 16kHz (hard requirement for Anam)
- Check avatar session logs for errors

### Video Not Appearing
- Grant camera permissions in browser
- Check `enableVision` is true in `app-config.ts`
- Verify user published video track to room

### Mobile Issues
- Check viewport meta tag in `layout.tsx`
- Test with browser DevTools mobile emulation
- Verify touch target sizes (minimum 44x44px)

## Performance Optimization

### Video Streaming
- Adaptive frame sampling: 1fps while speaking, 0.1fps when silent
- Optional: Enable `navigator.connection` detection for mobile bandwidth throttling
- H.264 codec preferred for better mobile compatibility

### Frontend Bundling
- Next.js automatic code-splitting and lazy loading
- Optimized imports from LiveKit components
- CSS-in-JS via Tailwind for minimal bundle

## API Reference

### RPC Methods (Backend → Frontend)

```typescript
// Update form field
room.sendRpc("updateField", { fieldId: string, value: string })

// Get current form state
room.sendRpc("getFormState", {}) → FormState

// Submit form data
room.sendRpc("submitForm", { [fieldId]: string }) → { success: boolean }
```

### RPC Methods (Frontend → Backend)

```typescript
// Get agent status
room.sendRpc("getAgentStatus", {}) → { state: "idle" | "speaking" | "listening" }
```

## Contributing

This is part of the LiveKit Agents framework. For development:

1. Create feature branch: `git checkout -b feature/real-time-video`
2. Make changes and test locally
3. Run type checks: `mypy src/agent.py` (backend), `tsc --noEmit` (frontend)
4. Submit PR with description

## License

Same as LiveKit Agents project (Apache 2.0)
