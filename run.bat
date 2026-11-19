@echo off
rem Start Calibration Tool (Windows). Run install.bat first.
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo venv not found. Run install.bat first.
    pause
    exit /b 1
)
cd intrinsic
"%~dp0venv\Scripts\python.exe" -m app.main %*
if errorlevel 1 pause
