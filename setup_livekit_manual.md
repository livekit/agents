# LiveKit Setup Guide

## Option 1: LiveKit Cloud (Easiest - Recommended)

### Step 1: Sign Up
1. Go to https://cloud.livekit.io
2. Sign up for a free account (free tier available)
3. Verify your email if required

### Step 2: Create a Project
1. After logging in, click "Create Project" or go to your dashboard
2. Give your project a name (e.g., "FillerFilterTest")
3. Select a region close to you
4. Click "Create"

### Step 3: Get Your Credentials
1. In your project dashboard, you'll see:
   - **WebSocket URL**: Something like `wss://your-project.livekit.cloud`
   - **API Key**: A long string (starts with something like `API...`)
   - **API Secret**: Another long string

2. Copy these three values

### Step 4: Set Environment Variables (Windows PowerShell)

**Option A: For Current Session Only**
```powershell
$env:LIVEKIT_URL = "wss://your-project.livekit.cloud"
$env:LIVEKIT_API_KEY = "your-api-key-here"
$env:LIVEKIT_API_SECRET = "your-api-secret-here"
```

**Option B: Add to .env file (Recommended)**
Create or edit a `.env` file in the project root (`C:\Users\sheet\agents\.env`):

```
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your-api-key-here
LIVEKIT_API_SECRET=your-api-secret-here
```

The demo agent already loads `.env` files automatically via `load_dotenv()`.

**Option C: Make Permanent in PowerShell**
Add to your PowerShell profile:
```powershell
# Open profile
notepad $PROFILE

# Add these lines:
$env:LIVEKIT_URL = "wss://your-project.livekit.cloud"
$env:LIVEKIT_API_KEY = "your-api-key-here"
$env:LIVEKIT_API_SECRET = "your-api-secret-here"
```

### Step 5: Run the Agent
```powershell
python examples/voice_agents/filler_filter_demo.py dev
```

---

## Option 2: Local LiveKit Server (Docker)

If you prefer to run LiveKit locally:

### Step 1: Install Docker
- Download Docker Desktop for Windows: https://www.docker.com/products/docker-desktop
- Install and start Docker Desktop

### Step 2: Run LiveKit Server
```powershell
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp `
  -e LIVEKIT_KEYS="devkey: devsecret" `
  livekit/livekit-server --dev
```

### Step 3: Set Environment Variables
```powershell
$env:LIVEKIT_URL = "ws://localhost:7880"
$env:LIVEKIT_API_KEY = "devkey"
$env:LIVEKIT_API_SECRET = "devsecret"
```

### Step 4: Run the Agent
```powershell
python examples/voice_agents/filler_filter_demo.py dev
```

---

## Quick Test

After setting up credentials, verify they're set:

```powershell
# Check if variables are set
echo $env:LIVEKIT_URL
echo $env:LIVEKIT_API_KEY
echo $env:LIVEKIT_API_SECRET
```

Then run:
```powershell
python examples/voice_agents/filler_filter_demo.py dev
```

You should see the agent start without the "ws_url is required" error.

---

## Troubleshooting

**Error: "ws_url is required"**
- Make sure you set all three variables: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- Check for typos in the variable names
- If using `.env` file, make sure it's in the project root

**Error: "Connection failed"**
- For LiveKit Cloud: Check your URL format (should start with `wss://`)
- For local server: Make sure Docker is running and the server container is active
- Check your internet connection

**Error: "Authentication failed"**
- Verify your API key and secret are correct
- Make sure there are no extra spaces when copying credentials

