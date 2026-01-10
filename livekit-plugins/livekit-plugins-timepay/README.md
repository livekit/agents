# TimePay AI TTS Plugin for LiveKit Agents

Hyper-realistic text-to-speech with support for Indian languages and emotion tags.

## Installation

```bash
pip install livekit-plugins-timepay
```

## Authentication

Set `TIMEPAY_API_KEY` in your `.env` file ([get one here](https://timepay.ai/)).

## Usage

Use TimePay TTS within an `AgentSession` or as a standalone speech generator.

```python
from livekit.plugins import timepay

tts = timepay.TTS()
```

Or with options:

```python
from livekit.plugins import timepay

tts = timepay.TTS(
    voice_id="Ogbs15oBevLzXsUuTtA1",  # Kartik (default)
    language="en",                    # en, hi, mr, ta, te, gu, kn, ml, bn, pa, od, as
    sample_rate=24000,                # 8000, 16000, or 24000 Hz
    speed=1.0,                        # 0.5 to 2.0
    add_wav_header=True,              # Add WAV header for playback
)
```

## Available Voices

| Voice ID | Name | Gender | Languages |
|----------|------|--------|-----------|
| `Ogbs15oBevLzXsUuTtA1` | Kartik | Male | English, Hindi, Marathi |
| `Owbs15oBevLzXsUurdA_` | Rahul | Male | English, Hindi |
| `PAbs15oBevLzXsUu4dCi` | Nisha | Female | English, Hindi |
| `PQbt15oBevLzXsUuNtD3` | Tulsi | Female | English, Hindi |
| `Pgbt15oBevLzXsUubdA6` | Seema | Female | English, Hindi |

## Emotion Tags

TimePay supports emotion tags to control the tone of speech:

```python
from livekit.plugins import timepay

tts = timepay.TTS()

# Emotion examples
emotion_text = "<angry>I'm so frustrated with this delay right now.</angry>"
whisper_text = "<whisper>Don't make a sound, they might hear us.</whisper>"
excited_text = "<excited>I can't believe we actually won!</excited>"
sad_text = "<sad>It's hard to say goodbye after all this time.</sad>"

# Synthesize with emotion
audio_stream = tts.synthesize(emotion_text)
```

Supported emotions:
- `<angry>` - Angry tone
- `<shouting>` - Shouting tone
- `<laughing>` - Laughing tone
- `<sad>` - Sad tone
- `<whisper>` - Whisper tone
- `<excited>` - Excited tone
- `<confused>` - Confused tone
- Neutral (no tags)

## Multi-Language Support

TimePay excels at Indian language support with native script handling:

```python
from livekit.plugins import timepay

# Hindi with English code-mixing
hindi_text = "आपका payment अभी तक नहीं आया, please check करें"

# Tamil with English
tamil_text = "நேத்து call பண்ணேன், but you didn't pick up"

# Complex multi-script example
multi_text = "நேத்து call பண்ணினா, ਉਹने pick ही नहीं कीता"

tts = timepay.TTS(language="hi")  # Set primary language
audio_stream = tts.synthesize(hindi_text)
```

## Supported Languages

| Code | Language | Script | Example |
|------|----------|--------|---------|
| `en` | English | Latin | Hello, how are you? |
| `hi` | Hindi | Devanagari | नमस्ते, आप कैसे हैं? |
| `mr` | Marathi | Devanagari | नमस्कार, तुम्ही कसे आहात? |
| `ta` | Tamil | Tamil | வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்? |
| `te` | Telugu | Telugu | నమస్కారం, మీరు ఎలా ఉన్నారు? |
| `gu` | Gujarati | Gujarati | નમસ્તે, તમે કેમ છો? |
| `kn` | Kannada | Kannada | ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ? |
| `ml` | Malayalam | Malayalam | നമസ്കാരം, നിങ്ങൾ എങ്ങനെയാണ്? |
| `bn` | Bengali | Bengali | নমস্কার, আপনি কেমন আছেন? |
| `pa` | Punjabi | Gurmukhi | ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ? |
| `od` | Odia | Odia | ନମସ୍କାର, ଆପଣ କେମିତି ଅଛନ୍ତି? |
| `as` | Assamese | Assamese | নমস্কাৰ, আপুনি কেনেকুৱা আছে? |

## Nonverbal Sounds

Insert non-speech sounds using special tags:

```python
# Nonverbal examples
cough_text = "I think [cough] I'm coming down with a cold."
sigh_text = "Alright [sigh], let's try this one more time."
sniffle_text = "It's just so beautiful... [sniffle]"
yawn_text = "I stayed up way too [yawn] late last night."
```

## Speed Control

Control pacing with speed tags:

```python
slow_text = "<slow>The secret code is... A... B... C...</slow>"
fast_text = "<fast>Terms and conditions apply void where prohibited.</fast>"
```

## Streaming

TimePay TTS supports HTTP streaming for real-time synthesis:

```python
from livekit.plugins import timepay

tts = timepay.TTS(voice_id="Ogbs15oBevLzXsUuTtA1")

# Create a stream for real-time synthesis
stream = tts.stream()

# Push text incrementally
stream.push_text("Hello, ")
stream.push_text("how are you today?")
stream.flush()  # Flush any remaining buffered text
stream.end_input()  # Signal end of input

# Consume audio as it's generated
async for audio in stream:
    # Process audio frames
    pass
```

## Complete Agent Example

```python
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import timepay, silero

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    agent = Agent(
        instructions="You are a friendly voice assistant that speaks multiple Indian languages."
    )
    
    session = AgentSession(
        vad=silero.VAD.load(),
        tts=timepay.TTS(
            voice_id="Ogbs15oBevLzXsUuTtA1",  # Kartik
            language="en",
            sample_rate=24000
        )
    )
    
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

## API Reference

### TTS Class

```python
class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: str = "Ogbs15oBevLzXsUuTtA1",
        language: str = "en",
        sample_rate: int = 24000,
        speed: float = 1.0,
        add_wav_header: bool = True,
        api_key: str | None = None,
        base_url: str = "https://api.tts.timepay.ai/api/v1",
        http_session: aiohttp.ClientSession | None = None,
    )
```

### Methods

- `synthesize(text: str)` - Convert text to speech
- `stream()` - Create a streaming synthesis session
- `list_voices()` - List available voices
- `aclose()` - Close the session and cleanup resources

## Best Practices

1. **Text Formatting**: Write numbers in words ("one two three" instead of "123")
2. **Language Mixing**: Use native scripts for Indian languages, Latin for English
3. **Emotion Tags**: Wrap entire sentences in emotion tags
4. **Sample Rate**: Use 24000 Hz for best quality
5. **Speed Control**: Keep speed between 0.5 and 2.0 for natural speech

## Resources

- [TimePay AI Documentation](https://docs.timepay.ai/)
