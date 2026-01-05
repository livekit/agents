# Testing Guide for Camb.ai Plugin

## Quick Start Testing

### 1. Set up your API key
```bash
export CAMB_API_KEY=your_camb_api_key_here
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Run example scripts

**Simple demo** (quickest test):
```bash
cd livekit-plugins/livekit-plugins-camb
uv run python examples/simple_demo.py
```

**Comprehensive test suite**:
```bash
cd livekit-plugins/livekit-plugins-camb
uv run python examples/test_camb_tts.py
```

This will test:
- ✅ Voice listing (`list_voices()`)
- ✅ Basic synthesis
- ✅ MARS models (mars-8, mars-8-flash, mars-8-instruct)
- ✅ Speed control
- ✅ Multi-language support (optional)

## Code Quality Tests

### Format check
```bash
uv run ruff format livekit-plugins/livekit-plugins-camb
```

### Lint check
```bash
uv run ruff check livekit-plugins/livekit-plugins-camb
```

### Type check
```bash
uv run mypy livekit-plugins/livekit-plugins-camb/livekit/plugins/camb
```

### All quality checks
```bash
make check
```

## Repository Test Suite

The LiveKit Agents repo has comprehensive integration tests for TTS plugins.

### Run TTS tests with Docker

The repo uses Docker Compose for integration testing:

```bash
cd tests
make test PLUGIN=camb
```

This will:
1. Start Docker containers (with toxiproxy for network testing)
2. Run pytest with the Camb.ai plugin
3. Test various scenarios (basic synthesis, fallback, error handling)
4. Clean up containers

### Run tests manually (without Docker)

```bash
# From repo root
uv run pytest tests/test_tts.py -k camb -v
```

### Test configuration

Tests are configured in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
addopts = ["--import-mode=importlib", "--ignore=examples"]
```

## Adding Camb to Integration Tests

The main TTS test file is at `tests/test_tts.py`. To include Camb.ai in automated tests:

1. **Import the plugin** (line 21-38 in test_tts.py):
```python
from livekit.plugins import (
    # ... other plugins
    camb,  # Add this
)
```

2. **Add to test matrix** (find the `@pytest.mark.parametrize` decorator):
```python
@pytest.mark.parametrize(
    "tts_cls",
    [
        # ... other plugins
        camb.TTS,  # Add this
    ],
)
```

3. **Set environment variable**:
```bash
export CAMB_API_KEY=your_key
```

## What Gets Tested

### Unit Tests (examples/)
- Plugin initialization
- Voice listing
- Basic synthesis
- Model selection (MARS variants)
- Speed control
- Language selection
- Error handling

### Integration Tests (tests/test_tts.py)
The main test suite checks:
- ✅ Basic synthesis with real API
- ✅ Audio quality (WER - Word Error Rate)
- ✅ Streaming behavior
- ✅ Error handling (network failures, timeouts)
- ✅ Fallback mechanisms
- ✅ Connection pooling
- ✅ Retry logic

### Performance Tests
- Time to first byte (TTFB)
- Total synthesis duration
- Audio duration vs text length
- Network resilience (via toxiproxy)

## Expected Test Results

### Basic Synthesis
- Should generate valid WAV audio
- Sample rate: 24000 Hz
- Channels: 1 (mono)
- Format: PCM 16-bit (audio/wav)

### Voice Listing
- Should return list of VoiceInfo objects
- Each voice has: id (int), name (str), gender, language

### MARS Models
- `mars-8`: Default, balanced
- `mars-8-flash`: Faster, similar quality
- `mars-8-instruct`: Supports user_instructions parameter

### Error Cases
- Invalid API key → APIStatusError (401)
- Invalid voice ID → APIStatusError (404)
- Network timeout → APITimeoutError
- Connection failure → APIConnectionError

## Troubleshooting

### "uv: command not found"
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### "CAMB_API_KEY not set"
```bash
export CAMB_API_KEY=your_key_here
```

### Import errors
Make sure you're running from the correct directory:
```bash
cd livekit-plugins/livekit-plugins-camb
uv run python examples/simple_demo.py
```

### Docker tests failing
```bash
cd tests
make down  # Clean up old containers
make up    # Start fresh
make test PLUGIN=camb
```

## CI/CD Integration

When ready for PR, the GitHub Actions workflow will run:
1. Ruff formatting check
2. Ruff linting
3. Mypy type checking
4. Pytest integration tests (if CAMB_API_KEY is set in secrets)

See `.github/workflows/tests.yml` and `.github/workflows/ci.yml` for details.
