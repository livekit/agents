# Frontend: Real-Time Vision + Avatar Agent

Next.js responsive web interface for real-time AI agent with video vision and animated avatars.

## Features

- **Mobile-First Responsive Design**: Optimized for all screen sizes (320px to 4K)
- **Real-time Streaming**: WebRTC video and audio via LiveKit
- **Interactive UI**: Avatar panel, chat transcript, and form controls
- **Type-Safe**: Full TypeScript support with strict type checking
- **Dark Mode**: Automatic light/dark theme with Tailwind CSS

## Quick Start

### Prerequisites

- Node.js 18+
- Backend agent running (`../agent-py/`)
- LiveKit server on `ws://localhost:7880`

### Installation

```bash
npm install
# or
pnpm install
```

### Configuration

Create `.env.local`:

```bash
# LiveKit Configuration (loaded by backend API route)
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
npm run build
npm start
```

## Architecture

### Component Hierarchy

```
layout.tsx
└── page.tsx
    └── AgentSessionProvider
        └── AppView
            ├── AvatarPanel
            ├── ChatPanel
            └── FormPanel
```

### Key Hooks

- **useAgentSession()**: Manages session state and connection
- **useRpcHandlers()**: Registers RPC methods for form communication
- **useVideoTrack()**: Subscribes to video streams
- **useAudioTrack()**: Manages audio streaming

## File Structure

```
frontend/
├── app/
│   ├── layout.tsx              # Root layout with HTML setup
│   ├── page.tsx                # Home page entry
│   └── api/
│       └── connection-details/
│           └── route.ts        # JWT token generation endpoint
├── components/
│   ├── app/
│   │   ├── agent-session-provider.tsx    # Session context
│   │   └── app-view.tsx                  # Main layout logic
│   └── ui/                               # Reusable UI components
├── hooks/
│   ├── useRpcHandlers.ts       # RPC method registration
│   └── useVideoTrack.ts         # Video track subscriptions
├── lib/
│   ├── app-config.ts           # Feature flags and config
│   ├── utils.ts                # LiveKit connection utilities
│   └── rpc-types.ts            # RPC method type definitions
├── styles/
│   └── globals.css             # Global Tailwind styles
├── public/                      # Static assets
├── package.json                # Dependencies
├── next.config.ts              # Next.js configuration
├── tsconfig.json               # TypeScript configuration
├── tailwind.config.ts          # Tailwind configuration
└── postcss.config.mjs          # PostCSS configuration
```

## Responsive Design

### Mobile-First Approach

**Mobile Layout (<768px):**
```
┌─────────────────┐
│                 │
│  Avatar Panel   │  Full width, 50% height
│  (Video)        │
│                 │
├─────────────────┤
│  Chat Transcript│  Full width, 50% height
│  Form Controls  │
└─────────────────┘
```

**Desktop Layout (≥768px):**
```
┌─────────────────┬─────────────────┐
│                 │                 │
│  Avatar Panel   │  Chat Transcript│
│  (Video)        │  Form Controls  │
│                 │                 │
└─────────────────┴─────────────────┘
```

### Tailwind Breakpoints

| Breakpoint | Width | Usage |
|-----------|-------|-------|
| xs | 320px | Phones (extra small) |
| sm | 640px | Phones (landscape) |
| md | 768px | Tablets |
| lg | 1024px | Desktops |
| xl | 1280px | Large desktops |

### Touch-Friendly Interactions

- Minimum button size: 44x44px (iOS standard)
- Safe area insets for notches/home indicators
- Vertical scroll for chat on mobile
- No hover-only interactions

## Configuration

### App Configuration (`lib/app-config.ts`)

```typescript
interface AppConfig {
  // Features
  enableAvatar: boolean;      // Show avatar video
  enableVision: boolean;      // Enable vision mode
  enableForm: boolean;        // Show form controls
  enableChat: boolean;        // Show chat history

  // Connection
  agentName: string;
  roomName: string;

  // UI
  brandName: string;
  primaryColor: string;
  darkMode: "auto" | "light" | "dark";

  // Mobile
  enableMobileOptimizations: boolean;
  mobileVideoQuality: "low" | "medium" | "high";

  // Debug
  debug: boolean;
}
```

### Runtime Configuration

Modify in `lib/app-config.ts`:

```typescript
const defaultConfig: AppConfig = {
  enableAvatar: true,
  enableVision: true,
  enableForm: false,          // Set true to show form
  enableChat: true,
  mobileVideoQuality: "high", // or "medium", "low"
  // ...
};
```

## API Routes

### POST /api/connection-details

Generates JWT token for LiveKit connection.

**Request:**
```json
{
  "room": "realtime",
  "identity": "user-123"
}
```

**Response:**
```json
{
  "url": "ws://localhost:7880",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

## RPC Communication

### Backend → Frontend

```typescript
// Update form field
room.sendRpc("updateField", { fieldId: "name", value: "John" })

// Get form state
const state = await room.sendRpc("getFormState", {})

// Submit form
const result = await room.sendRpc("submitForm", { name: "John" })
```

### Frontend → Backend

```typescript
// Get agent status
const status = await room.sendRpc("getAgentStatus", {})
```

## Development

### Type Checking

```bash
tsc --noEmit
```

### Linting

```bash
npm run lint
```

### Format Code

```bash
npx prettier --write .
```

## Browser Compatibility

- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- Mobile browsers (iOS Safari, Chrome Android)

## Performance Tips

### Code Splitting

Next.js automatically splits code at route boundaries. Components are lazy-loaded on demand.

### Image Optimization

Use Next.js Image component for automatic optimization:

```tsx
import Image from "next/image";

<Image
  src="/avatar.jpg"
  width={300}
  height={300}
  alt="Avatar"
/>
```

### Video Optimization

- Adaptive frame rate based on network speed
- Hardware acceleration for video decoding
- Mobile quality reduction (check `mobileVideoQuality` setting)

## Troubleshooting

### White Screen on Load

1. Check browser console for errors
2. Verify backend is running and accessible
3. Check `.env.local` configuration
4. Try hard refresh (Cmd+Shift+R or Ctrl+Shift+R)

### Connection Failed

```
Error: Failed to connect to room
```

→ Verify LiveKit server is running:
```bash
livekit-server --dev
```

→ Check `NEXT_PUBLIC_LIVEKIT_URL` matches server address

### No Video Appearing

1. Grant camera permissions in browser settings
2. Check `enableVision` and `enableAvatar` in app-config.ts
3. Verify backend is publishing video tracks
4. Check browser DevTools Network tab for WebRTC connection

### Mobile Layout Issues

1. Check viewport meta tag in `layout.tsx`
2. Test with browser DevTools mobile emulation (F12 → Device Toolbar)
3. Verify minimum touch target sizes (44x44px)
4. Check safe area insets in `styles/globals.css`

## Building for Production

### Optimization Checklist

- [ ] Test on actual mobile devices (not just DevTools)
- [ ] Verify all API keys are set via environment variables
- [ ] Run build test: `npm run build`
- [ ] Test production build locally: `npm start`
- [ ] Check bundle size: `npm run build` (look for warnings)
- [ ] Test with slow network (DevTools → Throttling)

### Docker Deployment

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install --production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## Next Steps

1. Connect to running backend agent
2. Test video and audio streams
3. Customize form fields for your use case
4. Add additional UI components as needed
5. Deploy to hosting platform (Vercel, AWS, GCP, etc.)

## References

- [Next.js Documentation](https://nextjs.org/docs)
- [LiveKit Components React](https://github.com/livekit/components-react)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [TypeScript Documentation](https://www.typescriptlang.org/docs/)

## Support

For issues with LiveKit, see [LiveKit Documentation](https://docs.livekit.io/).
For Next.js issues, see [Next.js GitHub Issues](https://github.com/vercel/next.js/issues).
