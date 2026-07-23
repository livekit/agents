# Contributing to livekit-plugins-google-adk

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### 1. Clone and Install

```bash
# Navigate to the plugin directory
cd livekit-plugins/livekit-plugins-google-adk

# Create virtual environment (optional but recommended)
uv sync

```

### 2. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=livekit.plugins.google_adk

# Run specific test
pytest tests/test_llm.py::TestGoogleADKLLM::test_chat_streaming
```

### 3. Code Quality

```bash
# Format code
black .

# Lint
ruff check .

# Type check
mypy livekit/plugins/google_adk/
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write docstrings for public APIs
- Update tests for your changes

### 3. Test Your Changes

```bash
# Run tests
pytest

# Run with the example
cd examples
python basic_voice_agent.py dev
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test changes
- `refactor:` - Code refactoring

## Adding New Features

### Adding Tool Calling Support

1. Update `llm.py` to handle tool calls in SSE stream
2. Convert LiveKit `FunctionInfo` to ADK tool format
3. Parse tool call events from ADK
4. Return `ToolCall` choices in `ChatChunk`
5. Add tests for tool calling
6. Update README with tool calling example

### Adding Multi-Agent Support

1. Add agent selection logic
2. Support agent handoff events from ADK
3. Update documentation with multi-agent example

## Testing Guidelines

### Unit Tests

- Test all public methods
- Mock external dependencies (aiohttp, ADK server)
- Test error handling paths
- Test edge cases (empty messages, malformed SSE, etc.)

### Integration Tests

- Test against real ADK server (optional, requires setup)
- Test full voice agent flow
- Test tool calling end-to-end

### Example Test

```python
@pytest.mark.asyncio
async def test_new_feature():
    """Test description."""
    # Arrange
    adk_llm = GoogleADK(
        api_base_url="http://localhost:8000",
        app_name="test-app",
        user_id="test-user",
    )

    # Act
    result = await adk_llm.new_feature()

    # Assert
    assert result is not None
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Short description.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When this error occurs
    """
    pass
```

### Updating README

- Add examples for new features
- Update configuration options
- Add troubleshooting tips

## Contributing to LiveKit

This plugin is designed to be contributed to the [LiveKit Agents](https://github.com/livekit/agents) repository.

### Steps to Contribute Upstream

1. **Open an Issue** in LiveKit repo proposing the integration
2. **Wait for Feedback** from maintainers
3. **Fork** the LiveKit Agents repo
4. **Copy Plugin** to `livekit-plugins/livekit-plugins-google-adk/`
5. **Adapt Structure** to match LiveKit's plugin conventions
6. **Add Tests** following LiveKit's testing patterns
7. **Submit PR** with:
   - Plugin code
   - Tests
   - Documentation
   - Example in `examples/` directory

### LiveKit Contribution Requirements

- Sign Google CLA (if required)
- Follow LiveKit's code style
- Include comprehensive tests
- Update LiveKit documentation
- Provide example usage

See [LiveKit CONTRIBUTING.md](https://github.com/livekit/agents/blob/main/CONTRIBUTING.md) for details.

## Code Review Process

1. Submit pull request
2. Wait for CI checks to pass
3. Address review comments
4. Once approved, changes will be merged

## Questions?

- Open an issue for questions
- Check existing issues and discussions
- Review LiveKit documentation

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
