# ==============================================================================
# Temporal Threat Detector - 1-Click Offline Localhost Website (PowerShell)
# ==============================================================================
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  TEMPORAL ANOMALY & THREAT DETECTOR - 100% OFFLINE LOCALHOST WEBSITE" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[*] Checking Python runtime..." -ForegroundColor Yellow

try {
    $pyVer = python --version 2>&1
    Write-Host "[+] Found: $pyVer" -ForegroundColor Green
} catch {
    Write-Host "[-] Python is not detected in PATH." -ForegroundColor Red
    exit 1
}

Write-Host "[*] Launching browser to http://localhost:8501 ..." -ForegroundColor Yellow
Start-Process "http://localhost:8501"

Write-Host "[*] Starting Streamlit Offline Engine..." -ForegroundColor Yellow
python -m streamlit run dashboard/app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
