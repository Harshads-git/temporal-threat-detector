Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " TEMPORAL THREAT DETECTOR - 100% OFFLINE LOCALHOST LAUNCHER" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Starting local Streamlit dashboard at http://localhost:8501 ..." -ForegroundColor Yellow
Write-Host "Zero internet connection required. All ML and SHAP models run locally." -ForegroundColor Gray
Write-Host "=====================================================================" -ForegroundColor Cyan

Set-Location -Path $PSScriptRoot
python -m streamlit run dashboard/app.py --server.port 8501 --server.headless $false
