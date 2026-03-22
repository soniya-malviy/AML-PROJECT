#!/bin/bash

# Navigate to the correct directory
cd "$(dirname "$0")/AML-PROJECT" || exit

# Create the virtual environment
python3 -m venv venv

echo "Virtual environment 'venv' successfully created in AML-PROJECT."
echo ""
echo "To activate it, run:"
echo "source AML-PROJECT/venv/bin/activate"

# Optional: Install requirements if they exist
if [ -f "requirements.txt" ]; then
    echo ""
    echo "To install requirements, run:"
    echo "AML-PROJECT/venv/bin/pip install -r AML-PROJECT/requirements.txt"
fi
