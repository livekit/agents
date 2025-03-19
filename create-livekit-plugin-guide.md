# How to create a LiveKit Plugin

## 1. Set Up Your Plugin

### Copy Template
Copy the `livekit-plugins-minimal` folder to start your plugin.

### Update Metadata

**Edit `package.json`:**
- Change the `name` field.

**Edit `pyproject.toml`:**
- Update project metadata.
- Add your specific dependencies in the `dependencies` section.
- Set appropriate Python version requirements.

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

### Update GitHub Workflow Files
Ensure your plugin is included in the CI/CD pipeline:

1. Add your plugin to `.github/workflows/ci.yml` under the `type-check` job:
   ```yaml
   - name: Check Types
     run: |
       mypy --install-types --non-interactive \
            # ... existing plugins ...
            -p livekit.plugins.yourplugin
   ```

2. If your plugin needs specific environment variables for testing, update `.github/workflows/tests.yml`:
   ```yaml
   - name: Run tests
     env:
       # ... existing env vars ...
       YOURCOMPANY_API_KEY: ${{ secrets.YOURCOMPANY_API_KEY }}
   ```

### Update Installation Script
Add your plugin to `livekit-plugins/install_local.sh`:
```bash
pip install \
  # ... existing plugins ...
  "${SCRIPT_DIR}/livekit-plugins-yourplugin"
```

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

### Update Main README.md Tables
Add your plugin to the appropriate integration tables in the main `README.md`:

1. For STT plugins, add to the STT table:
   ```markdown
   | YourCompany | livekit-plugins-yourplugin | ✅/❌ | [yourplugin.STT()](link-to-docs) |
   ```

2. For TTS plugins, add to the TTS table:
   ```markdown
   | YourCompany | livekit-plugins-yourplugin | ✅/❌ | ✅/❌ | [yourplugin.TTS()](link-to-docs) |
   ```

3. For LLM plugins, add to the LLM table:
   ```markdown
   | YourCompany | livekit-plugins-yourplugin | [yourplugin.LLM()](link-to-docs) |
   ```

4. For other plugins, add to the "Other plugins" table.

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

## 7. Final Submission Checklist

✅ Plugin code is implemented and follows LiveKit patterns
✅ Tests are created and passing
✅ CI/CD workflows are updated
✅ Installation script is updated
✅ Documentation is thorough
✅ README.md tables are updated
✅ Environment variables are documented
✅ Secrets are added to GitHub repository (if needed)
✅ Changeset is created
✅ Code is formatted and passes linting checks
