@echo off
rem Calibration Tool command line. Example: cli.bat intrinsic --help
cd /d "%~dp0intrinsic"
"%~dp0venv\Scripts\python.exe" -m app.tool %*
