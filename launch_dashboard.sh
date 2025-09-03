#!/bin/bash

# F1 Teammate Qualifying Dashboard Launcher
# Simple shell script to launch the Streamlit app

echo "🚀 Launching F1 Teammate Qualifying Dashboard..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📁 Working directory: $SCRIPT_DIR"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found in current directory"
    echo "Make sure you're running this from the f1_teammate_qual directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment (.venv) not found"
    echo "Please run 'make venv && make install' first"
    exit 1
fi

echo "✅ Found app.py and virtual environment"
echo "🔧 Activating virtual environment..."

# Activate virtual environment and launch Streamlit
source .venv/bin/activate

echo "🚀 Starting Streamlit app..."
echo "🌐 The app will open in your browser at: http://localhost:8501"
echo "📱 If it doesn't open automatically, navigate to the URL above"
echo "⏹️  Press Ctrl+C to stop the app"
echo "--------------------------------------------------"

# Launch Streamlit
streamlit run app.py
