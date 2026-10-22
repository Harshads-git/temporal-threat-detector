@echo off
cd /d "%~dp0"
REM ==============================================================================
REM Temporal Threat Detector - 100% Offline Localhost Website Runner
REM Runs the modern interactive dashboard on http://localhost:8501
REM ==============================================================================
echo ================================================================================
echo  🛡️  TEMPORAL THREAT DETECTOR - OFFLINE LOCALHOST WEBSITE
echo ================================================================================
echo [*] Checking Python environment...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [-] Error: Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo [*] Starting Streamlit Website on http://localhost:8501...
echo [*] Mode: 100%% Offline Embedded Dual-Tier AI Engine + Interactive Testing Lab
echo.
python -m streamlit run dashboard/app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
pause
