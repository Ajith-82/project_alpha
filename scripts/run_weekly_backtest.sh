#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_weekly_backtest.sh [india|us]
# Suggested cron (Sundays):
#   0 8 * * 0 /path/to/scripts/run_weekly_backtest.sh us
#   0 9 * * 0 /path/to/scripts/run_weekly_backtest.sh india

MARKET=${1:-us}

# Get the directory where the script is located (scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Navigate to project root (one level up)
cd "$SCRIPT_DIR/.."

LOG_DIR="./logs"
DATE=$(date +%Y-%m-%d)
mkdir -p "$LOG_DIR" ./data/models

LOG_FILE="$LOG_DIR/backtest_${MARKET}_${DATE}.log"

echo "=== Weekly Backtest: $MARKET — $(date) ===" | tee "$LOG_FILE"

poetry run python src/project_alpha.py \
    --market "$MARKET" \
    --strict \
    --backtest \
    --walk-forward \
    --wf-train-months 12 \
    --wf-test-months 3 \
    --initial-capital 10000 \
    --risk-per-trade 0.01 \
    --atr-multiplier 2.0 \
    --max-positions 10 \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "Backtest completed successfully at $(date)" | tee -a "$LOG_FILE"
else
    echo "Backtest failed with exit code $EXIT_CODE at $(date)" | tee -a "$LOG_FILE"
fi

exit $EXIT_CODE
