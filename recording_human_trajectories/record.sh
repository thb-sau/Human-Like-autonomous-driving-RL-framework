#!/bin/bash

# Check if the script is being run from the correct directory
if [[ ! -f "./key_record.py" ]] || [[ ! -f "./key.py" ]]; then
    echo "Error: Please run this script from the directory containing key_record.py and key.py."
    exit 1
fi

# Run the first Python script
echo "Running key_record.py..."
python ./key_record.py &

# Record the process ID of the first script
KEY_RECORD_PID=$!

# Wait for 5 seconds
echo "Waiting for 5 seconds before running key.py..."
sleep 5

# Run the second Python script
echo "Running key.py..."
python ./key.py &

# Wait for both scripts to complete
echo "Waiting for both scripts to finish..."
wait $KEY_RECORD_PID

echo "Both scripts have finished."