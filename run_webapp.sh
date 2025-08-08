#!/bin/bash

echo "🎨 Starting Image to SVG Web App..."
echo ""
echo "Installing/updating dependencies..."
pip install -r requirements.txt

echo ""
echo "Starting Flask server..."
echo "Open your browser to http://localhost:5000"
echo ""

python app.py