#!/bin/bash
# This script creates the directory structure for all 8 research modules.

echo "--- 📂 Creating directory structure... ---"

MODULES=(
    "1_LOB_Microstructure_Simulator"
    "2_CFMM_Arbitrage_Pathfinder"
    "3_CostModel_Transaction_Economics"
    "4_Execution_Engine_Skeleton"
    "5_Forecasting_Kalman_Cointegration"
    "6_Online_Learning_Execution_Routing"
    "7_Risk_Sizing_Kelly_Tail_Risk"
    "8_MEV_Mempool_Simulation"
)

for dir in "${MODULES[@]}"; do
    mkdir -p "$dir"
    echo "Created directory: $dir"
done

echo "--- ✅ Directory structure complete. ---"
