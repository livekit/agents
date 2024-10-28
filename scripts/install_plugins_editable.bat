@echo off
setlocal

set "REPO_ROOT=%~dp0.."
set "UTILS_BAT=%CURR_DIR%_utils.bat"

if "%VIRTUAL_ENV%"=="" (
    echo You are not in a virtual environment.
    exit /b 1
)

call "%UTILS_BAT%" print_heading "Installing livekit plugins in editable mode"
call "%REPO_ROOT%\livekitenv\Scripts\pip" install -e "%REPO_ROOT%\livekit-plugins\livekit-plugins-azure" --config-settings editable_mode=strict
call "%REPO_ROOT%\livekitenv\Scripts\pip" install -e "%REPO_ROOT%\livekit-plugins\livekit-plugins-cartesia" --config-settings editable_mode=strict
call "%REPO_ROOT%\livekitenv\Scripts\pip" install -e "%REPO_ROOT%\livekit-plugins\livekit-plugins-deepgram" --config-settings editable_mode=strict
call "%REPO_ROOT%\livekitenv\Scripts\pip" install -e "%REPO_ROOT%\livekit-plugins\livekit-plugins-elevenlabs" --config-settings editable_mode=strict
call "%REPO_ROOT%\livekitenv\Scripts\pip" install -e "%REPO_ROOT%\livekit-plugins\livekit-plugins-google" --config-settings editable_mode=strict
call "%REPO_ROOT%\livekitenv\Scripts\pip" install -e "%REPO_ROOT%\livekit-plugins\livekit-plugins-minimal" --config-settings editable_mode=strict
call "%REPO_ROOT%\livekitenv\Scripts\pip" install -e "%REPO_ROOT%\livekit-plugins\livekit-plugins-nltk" --config-settings editable_mode=strict
call "%REPO_ROOT%\livekitenv\Scripts\pip" install -e "%REPO_ROOT%\livekit-plugins\livekit-plugins-openai" --config-settings editable_mode=strict
call "%REPO_ROOT%\livekitenv\Scripts\pip" install -e "%REPO_ROOT%\livekit-plugins\livekit-plugins-rag" --config-settings editable_mode=strict
call "%REPO_ROOT%\livekitenv\Scripts\pip" install -e "%REPO_ROOT%\livekit-plugins\livekit-plugins-silero" --config-settings editable_mode=strict
call "%REPO_ROOT%\livekitenv\Scripts\pip" install -e "%REPO_ROOT%\livekit-plugins\livekit-plugins-browser" --config-settings editable_mode=strict
call "%REPO_ROOT%\livekitenv\Scripts\pip" install -e "%REPO_ROOT%\livekit-plugins\livekit-plugins-llama-index" --config-settings editable_mode=strict

endlocal
