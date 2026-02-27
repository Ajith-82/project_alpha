#!/bin/bash

# Configuration
PROJECT_DIR="/opt/developments/project_alpha"
LOG_DIR="$PROJECT_DIR/logs"
DATE=$(date +%Y-%m-%d)
MARKET=$1

# Ensure market is specified
if [ -z "$MARKET" ]; then
    echo "Usage: $0 [india|us]"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_analysis_${MARKET}_${DATE}.log"

echo "Starting daily analysis for $MARKET at $(date)" | tee -a "$LOG_FILE"

# Navigate to project directory
cd "$PROJECT_DIR" || exit 1

# Activate poetry env and run
# Note: --cache forces data refresh, --screeners volatility runs the new categorized analysis
# Load pre-trained volatility model if available (avoids retraining from scratch each run)
MODEL_FILE="$PROJECT_DIR/data/models/volatility_${MARKET}.pkl"
MODEL_FLAG=""
if [ -f "$MODEL_FILE" ]; then
    MODEL_FLAG="--load-model $MODEL_FILE"
fi

poetry run python src/project_alpha.py \
    --market "$MARKET" \
    --screeners volatility \
    --cache \
    --consensus \
    $MODEL_FLAG >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Analysis completed successfully at $(date)" | tee -a "$LOG_FILE"
else
    echo "Analysis failed with exit code $EXIT_CODE at $(date)" | tee -a "$LOG_FILE"
fi
