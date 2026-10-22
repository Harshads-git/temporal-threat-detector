@echo off
REM ==============================================================================
REM Temporal Threat Detector - 1-Click Offline Localhost Website Launcher
REM ==============================================================================
cd /d "%~dp0"
echo ================================================================================
echo   TEMPORAL ANOMALY ^& THREAT DETECTOR - 100%% OFFLINE LOCALHOST WEBSITE
echo ================================================================================
echo.
echo [*] Checking local Python environment...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [-] Error: Python is not detected in your PATH.
    echo [*] Please ensure Python 3.10+ is installed and accessible.
    pause
    exit /b 1
)

echo [+] Python runtime verified.
echo [*] Mode: 100%% OFFLINE (In-Process ML Engine + Novelty SHAP TreeExplainer)
echo [*] Target URL: http://localhost:8501
echo [*] Starting Streamlit Web Application on localhost...
echo.

REM Automatically pop browser when port becomes active
start "" http://localhost:8501

REM Launch Streamlit using python -m to avoid PATH issues
python -m streamlit run dashboard/app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

pause
