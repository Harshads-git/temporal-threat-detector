#!/usr/bin/env bash
# ==============================================================================
# Temporal Anomaly & Threat Detector - Single Command Execution Script
# ==============================================================================
set -e

echo "================================================================================"
echo " Starting Temporal Anomaly & Threat Detector Pipeline (Demo Mode)"
echo "================================================================================"

# Verify Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[-] Python is required but not installed. Exiting."
    exit 1
fi

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "[*] Using Python: $($PYTHON_CMD --version)"

# Check and install requirements if needed
echo "[*] Verifying dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt --quiet || true

# Run test suite to guarantee demo readiness
echo "[*] Executing test suite for validation..."
$PYTHON_CMD -m pytest tests/ -v

echo "[*] Generating synthetic baseline telemetry stream..."
$PYTHON_CMD -m data.synthetic_generator --count 200 --attack none

echo "================================================================================"
echo "[+] Pipeline components ready for live presentation and viva evaluation!"
echo "================================================================================"
