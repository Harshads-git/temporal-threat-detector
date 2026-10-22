#!/usr/bin/env bash
# ==============================================================================
# Temporal Anomaly & Threat Detector - Cross-Platform POSIX Demo Runner
# Designed for instant 1-command startup during faculty presentations & vivas
# ==============================================================================
set -e

# Terminal Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}================================================================================${NC}"
echo -e "${CYAN} Starting Temporal Anomaly & Threat Detector Pipeline (POSIX Demo Mode)${NC}"
echo -e "${CYAN}================================================================================${NC}"

# 1. Resolve Python executable
PYTHON_BIN="python3"
if ! command -v python3 &> /dev/null; then
    if command -v python &> /dev/null; then
        PYTHON_BIN="python"
    else
        echo -e "${RED}[-] Error: Python 3 is not installed or not in PATH.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}[*] Python environment ready:${NC} $($PYTHON_BIN --version)"

# 2. Execute automated test suite to verify pipeline integrity
echo -e "${YELLOW}[*] Executing test suite for validation...${NC}"
$PYTHON_BIN -m pytest tests/ -v

# 3. Generate sample diurnal baseline telemetry
echo -e "${YELLOW}[*] Generating synthetic baseline telemetry stream...${NC}"
$PYTHON_BIN -m data.synthetic_generator --count 200 --attack none

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}[+] Pipeline verification complete. Ready for live demonstration!${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
echo -e "${CYAN}[*] To spin up the FastAPI Backend Service:${NC}"
echo -e "    $PYTHON_BIN -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo -e "${CYAN}[*] To spin up the Streamlit Live Dashboard:${NC}"
echo -e "    $PYTHON_BIN -m streamlit run dashboard/app.py"
echo ""
echo -e "${CYAN}[*] To spin up full Docker stack (LocalStack + API + Dashboard):${NC}"
echo -e "    docker compose -f docker/docker-compose.yml up --build -d"
echo -e "${GREEN}================================================================================${NC}"
