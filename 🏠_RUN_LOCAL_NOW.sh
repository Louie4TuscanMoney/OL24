#!/bin/bash

# ONTOLOGIC XYZ - RUN EVERYTHING LOCALLY (NO VERCEL NEEDED)
# This starts the system on your computer - access via http://localhost:3000

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🏠 ONTOLOGIC XYZ - RUN LOCAL SYSTEM"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "This will run everything on your computer (no Vercel deployment yet)"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# STEP 1: Start Autonomous Backend
echo "🚀 STEP 1/3: Starting Autonomous Backend..."
echo ""
bash 🚀_START_AUTONOMOUS_SYSTEM.sh
echo ""
echo "✅ Backend running (PID files in 5. Live System/)"
echo ""

sleep 2

# STEP 2: Install Dashboard Dependencies
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📦 STEP 2/3: Installing Dashboard Dependencies..."
echo ""
cd "5. Live System/dashboard_pro"

if [ ! -d "node_modules" ]; then
    echo "Installing npm packages..."
    npm install
else
    echo "✅ Dependencies already installed"
fi

echo ""

# STEP 3: Start Dashboard
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🌐 STEP 3/3: Starting Dashboard..."
echo ""
echo "🎯 Dashboard will open on: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

npm run dev

