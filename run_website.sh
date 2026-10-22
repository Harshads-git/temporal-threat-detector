#!/usr/bin/env bash
echo "====================================================================="
echo " TEMPORAL THREAT DETECTOR - 100% OFFLINE LOCALHOST LAUNCHER"
echo "====================================================================="
echo "Starting local Streamlit dashboard at http://localhost:8501 ..."
echo "Zero internet connection required. All ML and SHAP models run locally."
echo "====================================================================="

streamlit run dashboard/app.py --server.port 8501 --server.headless false
