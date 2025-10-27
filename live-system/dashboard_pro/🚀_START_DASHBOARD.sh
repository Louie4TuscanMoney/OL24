#!/bin/bash

# Quick dashboard launcher
# Fixes common issues automatically

clear

echo "🚀 Starting Ontologic XYZ Dashboard..."
echo ""

# Check if we're in the right directory
CURRENT_DIR=$(pwd)
EXPECTED_DIR="/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System/dashboard_pro"

if [ "$CURRENT_DIR" != "$EXPECTED_DIR" ]; then
    echo "📍 Wrong directory. Moving to dashboard folder..."
    cd "$EXPECTED_DIR"
fi

echo "✅ In correct directory"
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "⚠️  Dependencies not installed. Installing now..."
    npm install
    echo ""
fi

echo "✅ Dependencies ready"
echo ""

# Kill any existing process on port 5173
if lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 5173 in use. Killing old process..."
    lsof -ti:5173 | xargs kill -9 2>/dev/null
    sleep 2
fi

echo "✅ Port 5173 clear"
echo ""

echo "🚀 Starting dashboard..."
echo ""
echo "   Dashboard will open at: http://localhost:5173"
echo "   Login password: rwwc2018"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

npm run dev

