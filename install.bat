@echo off
rem Calibration Tool setup (Windows): creates venv\ and installs requirements.txt.  "install.bat --dev" adds pytest.
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    py -3 -m venv venv 2>nul || python -m venv venv
    if errorlevel 1 (
        echo Python 3.10 or newer is required. https://www.python.org/downloads/
        pause
        exit /b 1
    )
)
venv\Scripts\python -m pip install --upgrade pip
if "%~1"=="--dev" (
    venv\Scripts\python -m pip install -r requirements-dev.txt
) else (
    venv\Scripts\python -m pip install -r requirements.txt
)
echo.
echo Done. Start the tool with run.bat
pause
