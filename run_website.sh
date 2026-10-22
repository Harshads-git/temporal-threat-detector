#!/usr/bin/env bash
# ==============================================================================
# Temporal Threat Detector - 1-Click Offline Localhost Website (Linux/macOS)
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "  TEMPORAL ANOMALY & THREAT DETECTOR - 100% OFFLINE LOCALHOST WEBSITE"
echo "================================================================================"

if ! command -v python3 &> /dev/null; then
    echo "[-] Python 3 is not installed or not in PATH."
    exit 1
fi

echo "[*] Launching browser to http://localhost:8501 ..."
if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:8501" &> /dev/null &
elif command -v open &> /dev/null; then
    open "http://localhost:8501" &> /dev/null &
fi

echo "[*] Starting Streamlit Offline Engine..."
python3 -m streamlit run dashboard/app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
