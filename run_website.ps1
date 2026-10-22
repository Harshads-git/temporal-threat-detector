# ==============================================================================
# Temporal Threat Detector - 100% Offline Localhost Website Runner (PowerShell)
# Runs the modern interactive dashboard on http://localhost:8501
# ==============================================================================
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  🛡️  TEMPORAL THREAT DETECTOR - OFFLINE LOCALHOST WEBSITE" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "[*] Checking Python environment..." -ForegroundColor Gray

try {
    $pyVer = & python --version 2>&1
    Write-Host "[+] Python detected: $pyVer" -ForegroundColor Green
} catch {
    Write-Error "Python is not installed or not found in PATH."
    exit 1
}

Write-Host "[*] Launching Streamlit Website on http://localhost:8501..." -ForegroundColor Yellow
Write-Host "[*] Mode: 100% Offline Embedded Dual-Tier AI Engine + Interactive Testing Lab" -ForegroundColor Yellow
Write-Host ""

& python -m streamlit run dashboard/app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
