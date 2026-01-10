# Azure Voice Agent Setup Guide

This guide will help you set up and run the Azure-powered voice agent.

## Prerequisites

✅ Virtual environment created and activated
✅ Dependencies installed (livekit-agents, azure plugin)

## Required Azure Services

You need the following Azure services set up:

### 1. Azure Speech Services (for STT/TTS)

**Setup:**
1. Go to [Azure Portal](https://portal.azure.com)
2. Create a new **Cognitive Services** resource or **Speech Service** resource
3. Once created, go to **Keys and Endpoint**
4. Copy:
   - **KEY 1** (this is your `AZURE_SPEECH_KEY`)
   - **Region** (e.g., `eastus`, `westus2`)

**Documentation:** https://learn.microsoft.com/en-us/azure/ai-services/speech-service/

### 2. Azure OpenAI (for LLM)

**Setup:**
1. Go to [Azure Portal](https://portal.azure.com)
2. Create an **Azure OpenAI** resource
3. Once created, go to **Keys and Endpoint**
4. Copy:
   - **Endpoint** URL (e.g., `https://your-resource.openai.azure.com`)
   - **KEY 1** (this is your `AZURE_OPENAI_API_KEY`)
5. Go to **Azure OpenAI Studio** > **Deployments**
6. Deploy a model (e.g., `gpt-4o`, `gpt-4o-mini`)
7. Note your **deployment name**

**Documentation:** https://learn.microsoft.com/en-us/azure/ai-services/openai/

## Environment Setup

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and fill in your Azure credentials:**
   ```bash
   # Azure Speech Services
   AZURE_SPEECH_KEY=your_actual_key_here
   AZURE_SPEECH_REGION=eastus

   # Azure OpenAI
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
   AZURE_OPENAI_API_KEY=your_actual_key_here
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
   ```

## Running the Agent

### Option 1: Console Mode (Local Testing with Microphone)

This mode requires a LiveKit server. You can use:
- **LiveKit Cloud** (easiest): https://cloud.livekit.io (free tier available)
- **Self-hosted**: https://docs.livekit.io/home/self-hosting/deployment/

```bash
# Add LiveKit credentials to .env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

# Run in console mode, no livekit connection needed
./venv/Scripts/python azure_agent.py console
```

### Option 2: Development Mode (with LiveKit)

```bash
./venv/Scripts/python azure_agent.py dev
```

Then connect using:
- LiveKit Agents Playground: https://agents-playground.livekit.io/
- Any LiveKit client SDK

### Option 3: Production Mode

```bash
./venv/Scripts/python azure_agent.py start
```

## Customization

### Change TTS Voice

Edit `azure_agent.py` and modify the TTS initialization:

```python
tts=azure.TTS(voice="en-US-Ava:DragonHDLatestNeural"),  # HD voices for more realistic output
# or
tts=azure.TTS(voice="en-US-JennyNeural"),               # Non HD voice
```

Available voices: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts

### Change Azure OpenAI Model

In your `.env`, update:
```bash
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini  # For faster, cheaper responses
# or
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o       # For better quality
```

### Add Custom Functions

Add function tools to your agent in `azure_agent.py`:

```python
@function_tool
async def my_custom_function(
    self, context: RunContext, parameter: str
):
    """Description of what this function does.

    Args:
        parameter: Description of the parameter
    """
    # Your logic here
    return "result"
```

## Troubleshooting

### "Missing API keys" error
- Make sure `.env` file exists in the project root
- Check that all required environment variables are set
- Restart your terminal/IDE after updating `.env`

### "Speech service error"
- Verify your `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` are correct
- Check that your Speech Service is active in Azure Portal
- Ensure you're using the correct region

### "OpenAI deployment not found"
- Verify your deployment name matches exactly (case-sensitive)
- Make sure the model is deployed in Azure OpenAI Studio
- Check that your API key has access to the deployment

### No audio in console mode
- Make sure your microphone is connected and working
- Check system audio permissions
- Try running with `--verbose` flag for more logs

## Next Steps

- Explore the `examples/voice_agents/` directory for more advanced examples
- Check out the [LiveKit Agents documentation](https://docs.livekit.io/agents/)
- Join the [LiveKit community](https://livekit.io/join-slack)

## Cost Considerations

- **Azure Speech**: Pay per character (TTS) and per hour (STT)
- **Azure OpenAI**: Pay per token
- **LiveKit Cloud**: Free tier available, paid plans for production

Estimate costs: https://azure.microsoft.com/en-us/pricing/calculator/
