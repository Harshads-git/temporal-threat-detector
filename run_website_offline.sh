#!/usr/bin/env bash
# ==============================================================================
# Temporal Threat Detector - Offline Interactive Website Runner (1-Click)
# ==============================================================================
set -e
echo "================================================================================"
echo " Launching Temporal Threat Detector - Modern Offline Interactive Website"
echo "================================================================================"
echo "[*] Mode: 100% Offline Embedded Dual-Tier AI & XAI Engine"
echo "[*] Zero cloud or internet connection required."
echo "[*] Features: Threat Lab (Live Input vs Output), Speedometer Gauge, Polar Radar, SHAP Waterfall"
echo "================================================================================"
echo ""
python -m streamlit run dashboard/app.py --server.headless=false
