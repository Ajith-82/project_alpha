#!/bin/bash

# Set current working directory
set cwd = `pwd`

# Generate a timestamp
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Run your Python script using poetry
cd /data/release/project_alpha

LOG_FILE="./logs/project_alpha_us.log"

# Redirect stdout to the log file with a timestamp
{
  echo "Script started at: $TIMESTAMP"
  echo "------------------"
  poetry run python ./src/project_alpha.py --market us --strict
  echo "------------------"
  echo "Script finished at: $(date +'%Y-%m-%d %H:%M:%S')"
} >> "$LOG_FILE" 2>&1

cd $cwd
