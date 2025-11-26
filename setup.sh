#!/bin/bash

set -e

echo "Setting up FireworksAI Text-to-SQL Take-Home Assessment..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Please install uv first: https://github.com/astral-sh/uv"
    echo "Quick install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment and install dependencies
echo "Creating virtual environment and installing dependencies..."
uv venv
source .venv/bin/activate
uv pip install -e .

# Download the Chinook database
echo ""
echo "Downloading Chinook database..."
if [ -f "Chinook.db" ]; then
    echo "Removing existing Chinook.db..."
    rm Chinook.db
fi

curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db

if [ -f "Chinook.db" ]; then
    echo "Successfully created Chinook.db"
else
    echo "Error: Failed to create database"
    exit 1
fi

echo ""
echo "Setup complete! 🎉"
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment: source .venv/bin/activate"
echo "  2. Set your FIREWORKS_API_KEY environment variable"
echo "  3. Run your solution code"
echo ""