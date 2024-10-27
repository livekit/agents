@echo off
:: Install Livekit-agents

set "CURR_DIR=%~dp0"
set "REPO_ROOT=%~dp0.."
set "UTILS_BAT=%CURR_DIR%_utils.bat"

:main
call "%UTILS_BAT%" print_heading "Installing livekit-agents"

:: Ensure virtual environment and pip are available
if not exist "%REPO_ROOT%\livekitenv\Scripts\pip.exe" (
    echo ERROR: pip is not installed in the virtual environment.
    goto :eof
)

@REM call "%UTILS_BAT%" print_heading "Installing requirements.txt"
@REM call "%REPO_ROOT%\livekitenv\Scripts\pip" install --no-deps -r "%REPO_ROOT%\livekit-agents\requirements.txt"
@REM if %ERRORLEVEL% neq 0 (
@REM     echo Failed to install requirements.
@REM     goto :eof
@REM )

call "%UTILS_BAT%" print_heading "Installing livekit agents with [dev] extras"
call "%REPO_ROOT%\livekitenv\Scripts\pip" install --editable "%REPO_ROOT%\livekit-agents[dev]"
if %ERRORLEVEL% neq 0 (
    echo Failed to install livekit-agents.
    goto :eof
)

goto :eof
