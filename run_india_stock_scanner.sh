#!/bin/bash

# Generate a timestamp
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Define the path to your Python virtual environment
VIRTUALENV_PATH="/home/ajithkv/projects/StockScanner/myvenv"

# Activate the virtual environment
source "$VIRTUALENV_PATH/bin/activate"

# Run your Python script within the virtual environment
cd /home/ajithkv/projects/StockScanner

LOG_FILE="./logs/stock_scanner.log"

# Redirect stdout to the log file with a timestamp
{
  echo "Script started at: $TIMESTAMP"
  echo "------------------"
  python ./src/screenipy.py --options india:5:1
  echo "------------------"
  echo "Script finished at: $(date +'%Y-%m-%d %H:%M:%S')"
} >> "$LOG_FILE" 2>&1

# Deactivate the virtual environment when done (optional)
deactivate
