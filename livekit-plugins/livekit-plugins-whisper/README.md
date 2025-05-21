# Whisper plugin for LiveKit Agents

Provides Speech-to-Text (STT) functionality using OpenAI's Whisper models.
The Whisper models are run locally, and no API key is required.

## Installation

```bash
pip install livekit-plugins-whisper
```

## Basic Usage

Here's how to integrate the Whisper STT into your LiveKit Voice Agent:

```python
from livekit.agents import JobContext, JobType
from livekit.agents.voice_agent import VoiceAgent
from livekit.plugins.whisper import STT

# Example of how to use the STT component
async def entrypoint(ctx: JobContext):
    # Initialize the Whisper STT engine
    # By default, it uses the "base" model.
    # You can specify a different model, e.g., STT(model_name="tiny")
    stt_plugin = STT() 

    # Example VoiceAgent setup (assuming you have TTS and LLM components)
    # agent = VoiceAgent(stt=stt_plugin, tts=your_tts_plugin, llm=your_llm_plugin)
    
    # Start the agent
    # await agent.start(room=ctx.room)

    # Your agent logic would go here.
    # For STT, you would typically process audio input from the room.
    # The VoiceAgent class handles the STT stream internally when it receives audio.
    # Transcribed text can be accessed via events or callbacks provided by VoiceAgent.
    pass

# Example worker setup (if you are running this as part of a LiveKit Agent worker)
# if __name__ == "__main__":
#     from livekit.agents import cli
#     # Define your worker and add job types
#     # cli.run_app(JobType.VOICE, entrypoint) 
```

## Models

Whisper offers several models with varying sizes, performance, and resource requirements. The models are downloaded locally the first time they are used.

Available models include:
- `tiny`
- `base`
- `small`
- `medium`
- `large`
- `tiny.en`, `base.en`, `small.en`, `medium.en` (English-optimized versions)

You can specify the model when creating the `STT` instance:

```python
stt_small = STT(model_name="small")
stt_base_english = STT(model_name="base.en")
```

If no model is specified, `"base"` is used by default.

## Language Support

Whisper can automatically detect the language of the spoken audio, or you can specify a language code (e.g., `"en"`, `"es"`, `"fr"`) during initialization:

```python
stt_spanish = STT(language="es")
stt_autodetect = STT() # Language will be auto-detected
```

## Additional Information

- **Local Execution**: All processing happens locally. No data is sent to external APIs for STT.
- **No API Key**: Since models run locally, no API key is needed.
- **ffmpeg**: Whisper requires `ffmpeg` to be installed on your system. Please ensure it's available in your PATH.
  ```bash
  # On Debian/Ubuntu
  sudo apt update && sudo apt install ffmpeg
  # On macOS (using Homebrew)
  brew install ffmpeg
  # On Windows (using Chocolatey)
  choco install ffmpeg
  ```

For more details on the Whisper models and their capabilities, please refer to the official [OpenAI Whisper GitHub repository](https://github.com/openai/whisper).
