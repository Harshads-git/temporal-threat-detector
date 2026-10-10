@echo off
REM ==============================================================================
REM Temporal Anomaly & Threat Detector - Windows Runner Script
REM ==============================================================================
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
