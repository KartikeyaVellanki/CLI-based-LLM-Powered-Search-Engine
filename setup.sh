#!/bin/bash

# Check if virtual environment already exists
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
else
  echo "Virtual environment already exists. Skipping creation."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies without using cache and with PEP 517 support
pip install -r requirements.txt --no-cache-dir --use-pep517

echo "Setup completed successfully."
