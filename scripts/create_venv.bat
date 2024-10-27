:: create_venv.bat - Create a new virtual environment for livekit agents development
@echo off
setlocal EnableDelayedExpansion

set "CURR_DIR=%~dp0"
set "REPO_ROOT=%~dp0.."
set "VENV_DIR=%REPO_ROOT%\livekitenv"
set "UTILS_BAT=%CURR_DIR%_utils.bat"

:: Check for admin privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo WARNING: Script is not running with administrator privileges
    echo This might cause permission errors
    choice /C YN /M "Do you want to continue anyway"
    if errorlevel 2 exit /b 1
)

:: Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

call "%UTILS_BAT%" print_heading "livekit agents dev setup"

:: Check if VENV_DIR exists and virtual environment is not active
if defined VIRTUAL_ENV (
    echo Virtual environment is active. Deactivate before running this script.
    exit /b 1
)

:: Try to remove existing venv if it exists
if exist "%VENV_DIR%" (
    call "%UTILS_BAT%" print_heading "Removing existing venv: %VENV_DIR%"
    rd /s /q "%VENV_DIR%" 2>nul
    if exist "%VENV_DIR%" (
        echo WARNING: Could not remove existing virtual environment
        echo Please close any programs that might be using the virtual environment
        echo or remove the directory manually: %VENV_DIR%
        exit /b 1
    )
)

:: Create the virtual environment
call "%UTILS_BAT%" print_heading "Creating livekit venv: %VENV_DIR%"
python -m venv "%VENV_DIR%" 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to create virtual environment
    echo Please ensure you have write permissions to: %VENV_DIR%
    echo Try running the script as Administrator
    exit /b %ERRORLEVEL%
)

:: Verify venv creation
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo ERROR: Virtual environment creation failed
    echo Please check if you have sufficient disk space and permissions
    exit /b 1
)

call "%UTILS_BAT%" print_heading "Upgrading pip to the latest version"
call "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to upgrade pip
    echo Please run the script as Administrator or check your network connection
    exit /b %ERRORLEVEL%
)

call "%UTILS_BAT%" print_heading "Installing base python packages"
call "%VENV_DIR%\Scripts\pip" install pip-tools twine build
if %ERRORLEVEL% neq 0 (
    echo WARNING: First attempt to install packages failed. Retrying...
    timeout /t 5 /nobreak
    call "%VENV_DIR%\Scripts\pip" install pip-tools twine build
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to install required packages
        echo Please check your network connection and try again
        exit /b %ERRORLEVEL%
    )
)

:: Install workspace packages
call "%VENV_DIR%\Scripts\activate"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to activate virtual environment
    exit /b %ERRORLEVEL%
)

call "%CURR_DIR%install.bat"

call "%UTILS_BAT%" print_heading "Virtual environment created successfully!"
echo To activate, run: %VENV_DIR%\Scripts\activate

endlocal