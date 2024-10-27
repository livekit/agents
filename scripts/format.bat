@echo off
:: Formats livekit agents

set "CURR_DIR=%~dp0"
set "REPO_ROOT=%~dp0.."
set "UTILS_BAT=%CURR_DIR%_utils.bat"

:main
call "%UTILS_BAT%" print_heading "Formatting livekit agents"

:: Ensure virtual environment tools are available
if not exist "%REPO_ROOT%\livekitenv\Scripts\ruff.exe" (
    echo ERROR: Ruff is not installed in the virtual environment.
    goto :eof
)

if not exist "%REPO_ROOT%\livekitenv\Scripts\mypy.exe" (
    echo ERROR: Mypy is not installed in the virtual environment.
    goto :eof
)

if not exist "%REPO_ROOT%\livekitenv\Scripts\pytest.exe" (
    echo ERROR: Pytest is not installed in the virtual environment.
    goto :eof
)

call "%UTILS_BAT%" print_heading "Running: ruff format %REPO_ROOT%"
call "%REPO_ROOT%\livekitenv\Scripts\ruff" format "%REPO_ROOT%"
if %ERRORLEVEL% neq 0 (
    echo Failed to format with ruff.
    goto :eof
)

call "%UTILS_BAT%" print_heading "Running: ruff check %REPO_ROOT%"
call "%REPO_ROOT%\livekitenv\Scripts\ruff" check "%REPO_ROOT%"
if %ERRORLEVEL% neq 0 (
    echo Failed ruff check.
    goto :eof
)

call "%UTILS_BAT%" print_heading "Running: mypy %REPO_ROOT%"
call "%REPO_ROOT%\livekitenv\Scripts\mypy" "%REPO_ROOT%"
if %ERRORLEVEL% neq 0 (
    echo Failed mypy check.
    goto :eof
)

call "%UTILS_BAT%" print_heading "Running: pytest %REPO_ROOT%"
call "%REPO_ROOT%\livekitenv\Scripts\pytest" "%REPO_ROOT%"
if %ERRORLEVEL% neq 0 (
    echo Failed pytest.
    goto :eof
)

goto :eof
