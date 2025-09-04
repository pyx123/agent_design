#!/bin/bash
# Start script for DevOps Agent

set -e

# Change to project root
cd "$(dirname "$0")/.."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Initialize database
echo "Initializing database..."
python -c "
import asyncio
from src.services import init_database
asyncio.run(init_database())
"

# Start the application
echo "Starting DevOps Agent API..."
python -m src.app