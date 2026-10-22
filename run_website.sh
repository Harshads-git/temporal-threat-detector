#!/usr/bin/env bash
# ==============================================================================
# Temporal Threat Detector - 100% Offline Localhost Website Runner (Linux/macOS)
# Runs the modern interactive dashboard on http://localhost:8501
# ==============================================================================
set -e
echo "================================================================================"
echo " 🛡️  TEMPORAL THREAT DETECTOR - OFFLINE LOCALHOST WEBSITE"
echo "================================================================================"
echo "[*] Starting Streamlit Website on http://localhost:8501..."
echo "[*] Mode: 100% Offline Embedded Dual-Tier AI Engine + Interactive Testing Lab"
echo ""
python -m streamlit run dashboard/app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
