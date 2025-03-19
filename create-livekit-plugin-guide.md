
# How to create a LiveKit Plugin

## 1. Set Up Your Plugin

### Copy Template
Copy the `livekit-plugins-minimal` folder to start your plugin.

### Update Metadata

**Edit `package.json`:**
- Change the `name` field.

**Edit `pyproject.toml`:**
- Update project metadata.

### Adjust Directory Structure
1. Update plugin class and imports in `__init__.py`.
2. Adjust logger details in `log.py`.
3. Set your version number in `version.py`.
4. Create an empty `py.typed` file to enable type hints.

## 2. Implement Plugin Services

### Study Existing Examples
Check existing plugins for reference:
- **STT:** Deepgram, AssemblyAI, OpenAI
- **TTS:** Cartesia, ElevenLabs, Deepgram, AWS
- **LLM:** OpenAI, Anthropic, Google

### Create Service Files
Depending on your service, create:
- `stt.py`: Speech-to-text
- `tts.py`: Text-to-speech
- `llm.py`: Language models
- `models.py`: Shared data structures or enums
- `utils.py`: Utility functions

Match existing plugin class structures and methods.

## 3. Test Your Plugin

### Update Tests
Add your plugin to the appropriate tests:
- STT → `tests/test_stt.py`
- TTS → `tests/test_tts.py`
- LLM → `tests/test_llm.py`

### Execute Tests
Run the tests to validate your implementation.

## 4. Linting and Documentation

LiveKit uses `ruff` for linting and formatting:
```bash
ruff format .
ruff check .
```

### Update Your README
Clearly document:
- Plugin description
- Setup steps
- Example usage
- Required environment variables

### Environment Variables
Clearly define environment variables:
- Example: `YOURCOMPANY_API_KEY` for YourCompany API.

## 5. Submit Your Changes

### Create a Changeset
Document your plugin changes:
```bash
pnpm changeset add
```

### Publishing Steps
Before publishing:
1. Submit a pull request.
2. Confirm all tests pass.
3. Address code quality issues.
4. Ensure documentation is thorough.

## 6. Best Practices
- Follow existing plugin patterns.
- Handle API connection errors gracefully.
- Clearly name environment variables.
- Use complete type annotations.
- Document public classes and methods.
- Ensure full test coverage.
- Properly close resources and connections.
- Adhere strictly to base class interfaces.
