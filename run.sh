#!/bin/bash

# Set PYTHONPATH to include the src directory
export PYTHONPATH=$(pwd)/src

# Run the main script
python python inference.py -conf config/config.json -prob 0.9 -o 0.1 -ac 0 -au 0 -aip "127.0.0.1"
