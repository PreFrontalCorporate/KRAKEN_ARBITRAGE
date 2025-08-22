#!/bin/bash

# This script runs all 8 agents in parallel using nohup.
# Each agent will have its own log file in its directory.

echo "--- 🚀 Launching the Autonomous Research Swarm ---"

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

# First, ensure no old agent processes are lingering
echo "  > Stopping any previously running agent processes..."
pkill -f "python3 agent.py"
# Wait a moment to ensure processes have terminated
sleep 2

# Remove any stale lock file if it exists
rm -f .agent/run.lock

# Start each agent process
for dir in "${MODULES[@]}"; do
    PROMPT_FILE="${dir}/prompt.txt"
    LOG_FILE="${dir}/agent.log"

    if [ -f "$PROMPT_FILE" ]; then
        echo "  > Launching agent for: ${dir}"
        # The agent will operate inside its target directory.
        # We redirect stdout and stderr to its dedicated log file.
        nohup python3 agent.py "${PROMPT_FILE}" "${dir}" > "${LOG_FILE}" 2>&1 &
    else
        echo "  > WARNING: Prompt file not found for ${dir}. Skipping."
    fi

    # A small delay to prevent any potential startup race conditions
    sleep 1
done

echo ""
echo "--- ✅ All agents have been launched successfully! ---"
echo "You can now safely close this terminal."
echo ""
echo "--- 🕵️‍♂️ How to Monitor Progress ---"
echo "  - To see which agents are still running: pgrep -af 'agent.py'"
echo "  - To view a live stream of a specific agent's log:"
echo "    tail -f 2_CFMM_Arbitrage_Pathfinder/agent.log"
echo "  - To check the status of all agents at once:"
echo "    tail -n 5 */agent.log"
echo ""
echo "When an agent is finished, it will stop running and you will find the"
echo "final .ipynb and .pdf report inside its directory."
