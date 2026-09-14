@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE="
set "PYTHON_ARGS="
set "LOG_FILE=%~dp0ON_error.log"

if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON_EXE=py"
            set "PYTHON_ARGS=-3"
        )
    )

    if not defined PYTHON_EXE (
        python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON_EXE=python"
        )
    )
)

if not defined PYTHON_EXE (
    echo Camera Calibration Tool could not start.
    echo.
    echo Python was not found.
    echo Install Python 3.10 or newer, then run ON.bat again.
    echo.
    echo Recommended setup:
    echo   python -m venv venv
    echo   venv\Scripts\python -m pip install -r requirements.txt
    echo.
    echo This window will stay open so you can read the message.
    cmd /k
    exit /b 1
)

"%PYTHON_EXE%" %PYTHON_ARGS% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo Camera Calibration Tool could not start.
    echo.
    echo Python 3.10 or newer is required, but the selected Python failed the version check.
    echo Selected Python:
    echo   "%PYTHON_EXE%" %PYTHON_ARGS%
    echo.
    echo Install Python 3.10 or newer, then run ON.bat again.
    echo.
    echo This window will stay open so you can read the message.
    cmd /k
    exit /b 1
)

"%PYTHON_EXE%" %PYTHON_ARGS% -m app.main > "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Camera Calibration Tool exited with an error: %EXIT_CODE%
    echo Error log:
    echo   %LOG_FILE%
    echo If this is the first run, install dependencies with:
    echo   "%PYTHON_EXE%" %PYTHON_ARGS% -m pip install -r requirements.txt
    echo.
    echo This window will stay open so you can read the message.
    cmd /k
)

exit /b %EXIT_CODE%
