#!/bin/bash

# Set current working directory
set cwd = `pwd`

# Generate a timestamp
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Define the path to your Python virtual environment
VIRTUALENV_PATH="/data/release/project_alpha/myenv"

# Activate the virtual environment
source "$VIRTUALENV_PATH/bin/activate"

# Run your Python script within the virtual environment
cd /data/release/project_alpha

LOG_FILE="./logs/project_alpha_india.log"

# Redirect stdout to the log file with a timestamp
{
  echo "Script started at: $TIMESTAMP"
  echo "------------------"
  python ./src/project_alpha.py --market india
  echo "------------------"
  echo "Script finished at: $(date +'%Y-%m-%d %H:%M:%S')"
} >> "$LOG_FILE" 2>&1

# Deactivate the virtual environment when done (optional)
deactivate

cd $cwd