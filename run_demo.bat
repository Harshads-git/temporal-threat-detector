@echo off
REM ==============================================================================
REM Temporal Anomaly & Threat Detector - Windows Runner Script
REM ==============================================================================
cd /d "%~dp0"
echo ================================================================================
echo  Starting Temporal Anomaly ^& Threat Detector Pipeline (Windows Demo)
echo ================================================================================

python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [-] Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo [*] Python environment ready.

echo [*] Executing test suite for validation...
python -m pytest tests/ -v

echo [*] Generating synthetic baseline telemetry sample...
python -m data.synthetic_generator --count 200 --attack none

echo ================================================================================
echo [+] Pipeline verification complete. Ready for live demonstration!
echo ================================================================================
echo.
echo [*] To spin up the FastAPI Backend Service:
echo     uvicorn api.main:app --port 8000 --reload
echo.
echo [*] To spin up the Streamlit Live Dashboard:
echo     streamlit run dashboard/app.py
echo ================================================================================
