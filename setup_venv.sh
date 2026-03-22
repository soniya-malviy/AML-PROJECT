#!/bin/bash

# Navigate to the directory containing this script (assumed repo root)
cd "$(dirname "$0")" || exit

# Create the virtual environment
python3 -m venv venv

echo "Virtual environment 'venv' successfully created in the current directory."
echo ""
echo "To activate it, run:"
echo "source venv/bin/activate"

# Optional: Install requirements if they exist
if [ -f "requirements.txt" ]; then
    echo ""
    echo "To install requirements, run:"
    echo "venv/bin/pip install -r requirements.txt"
fi
