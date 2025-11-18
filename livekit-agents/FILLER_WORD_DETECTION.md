# Filler Word Detection for LiveKit Agents

This feature enhances LiveKit agents with the ability to detect and handle filler words in real-time speech processing. It's particularly useful for creating more natural voice interactions by intelligently ignoring filler words when the agent is speaking while still recognizing them as valid speech when the agent is silent.

## Features

- **Phonetic Matching**: Supports multiple phonetic algorithms (Soundex, Metaphone, Double Metaphone) to detect variations of filler words
- **Configurable Detection**: Customize which words are considered fillers and how they're detected
- **Real-time Processing**: Works with LiveKit's real-time audio streams
- **Agent Awareness**: Intelligently handles filler words based on whether the agent is currently speaking
- **Dynamic Updates**: Update filler words and phonetic configurations at runtime

## Installation

1. Install the required dependencies:

```bash
pip install jellyfish>=1.0.0
```

## Usage

### Basic Usage

```python
from livekit.agents import vad
from livekit.agents.voice import VADWrapper, FillerWordConfig

# Create a VAD instance (e.g., from LiveKit's inference module)
vad_instance = vad.VAD.from_model_string("silero-vad")

# Configure filler word detection
filler_config = FillerWordConfig(
    filler_words=["um", "uh", "like", "you know"],
    case_sensitive=False,
    word_boundary=True,
    phonetic_config={
        "enabled": True,
        "algorithm": "double_metaphone",
        "min_word_length": 1,
        "include_original": True
    }
)

# Create the VAD wrapper
vad_wrapper = VADWrapper(vad_instance, VADWrapperConfig(filler_config=filler_config))

# Use the wrapper in your agent
# ...

# Update agent speaking state
vad_wrapper.set_agent_speaking(True)  # Filler words will now be filtered
# ...
vad_wrapper.set_agent_speaking(False)  # All speech will be processed
```

### Updating Filler Words at Runtime

```python
# Add new filler words
vad_wrapper.update_filler_words(["um", "uh", "like", "you know", "basically"])

# Add custom phonetic mappings
vad_wrapper.add_phonetic_mapping("hmm", ["hmmm", "hm", "hmmmm"])

# Update phonetic configuration
vad_wrapper.update_phonetic_config(enabled=True, algorithm="metaphone")
```

## Configuration

### FillerWordConfig

- `filler_words`: List of filler words to detect (default: `["uh", "umm", "hmm", "haan"]`)
- `case_sensitive`: Whether matching should be case-sensitive (default: `False`)
- `word_boundary`: Whether to match whole words only (default: `True`)
- `log_filtered`: Whether to log filtered filler words (default: `True`)
- `phonetic_config`: Configuration for phonetic matching (see below)
- `custom_phonetic_mappings`: Custom phonetic mappings for specific words

### PhoneticConfig

- `enabled`: Whether phonetic matching is enabled (default: `True`)
- `algorithm`: Phonetic algorithm to use ("soundex", "metaphone", or "double_metaphone") (default: "double_metaphone")
- `min_word_length`: Minimum word length to apply phonetic matching (default: `2`)
- `include_original`: Whether to include the original word in phonetic forms (default: `True`)
- `custom_mappings`: Custom phonetic mappings for specific words

## How It Works

1. When the agent is speaking, the VAD wrapper filters out any detected filler words from the speech stream.
2. When the agent is not speaking, all speech is passed through normally, including filler words.
3. Phonetic matching allows for detection of variations and mispronunciations of filler words.
4. The system is designed to be efficient, with caching of phonetic representations for better performance.

## Best Practices

1. **Start with Common Fillers**: Begin with common filler words in your target language.
2. **Use Phonetic Matching**: Enable phonetic matching to catch variations of filler words.
3. **Test with Real Data**: Test with real user speech to identify additional filler words to add to your configuration.
4. **Monitor Logs**: Keep an eye on the logs to identify any false positives or missed filler words.

## Example Integration with AgentSession

```python
from livekit.agents import AgentSession

class MyAgentSession(AgentSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Get the VAD wrapper
        self._vad_wrapper = self._vad  # If VAD was already wrapped
        
        # Or create and set it if not
        # from livekit.agents.voice import VADWrapper, FillerWordConfig
        # self._vad_wrapper = VADWrapper(self._vad)
        # self._vad = self._vad_wrapper
    
    async def say(self, text: str, **kwargs):
        # Set agent as speaking before starting speech
        if hasattr(self, '_vad_wrapper'):
            self._vad_wrapper.set_agent_speaking(True)
        
        try:
            # Your existing say implementation
            await super().say(text, **kwargs)
        finally:
            # Ensure we always set speaking back to False
            if hasattr(self, '_vad_wrapper'):
                self._vad_wrapper.set_agent_speaking(False)
```

## Troubleshooting

### Filler Words Not Being Detected

1. Check that phonetic matching is enabled if you're expecting variations of words to be detected.
2. Verify that the filler words in your configuration match the expected case sensitivity.
3. Check the logs for any warnings or errors related to the phonetic matching.

### Performance Issues

1. If experiencing performance issues, try increasing the `min_word_length` in the phonetic config.
2. Consider disabling phonetic matching if not needed for your use case.
3. Check that you're not adding an excessive number of filler words.

## License

This feature is part of the LiveKit Agents project and is licensed under the same terms.
