#!/bin/bash

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🔥 ONTOLOGIC XYZ - COMPLETE LIVE TRADING SYSTEM"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "This launches:"
echo "  1. Trading Dashboard API (port 8001)"
echo "  2. SolidJS Dashboard (port 3000)"
echo "  3. Live game monitoring"
echo "  4. OntoRisk integration"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if in correct directory
if [ ! -d "5. Live System" ]; then
    echo "❌ Error: Run this script from the ML Research root directory"
    exit 1
fi

cd "5. Live System"

# Check dependencies
echo "📋 Checking dependencies..."
echo ""

# Check Python dependencies
python3 -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ FastAPI not installed"
    echo "   Install with: pip install fastapi uvicorn"
    read -p "   Install now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install fastapi uvicorn requests beautifulsoup4
    fi
fi

# Check if dashboard node_modules exists
if [ ! -d "dashboard/node_modules" ]; then
    echo "📦 Installing dashboard dependencies..."
    cd dashboard
    npm install
    cd ..
    echo "✅ Dependencies installed"
    echo ""
fi

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 STARTING SYSTEM"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Start backend API in background
echo "🔥 Starting Trading Dashboard API (port 8001)..."
python3 trading_dashboard_api.py > logs/api.log 2>&1 &
API_PID=$!
echo "✅ API started (PID: $API_PID)"
echo ""

# Wait for API to be ready
echo "⏳ Waiting for API to be ready..."
sleep 3

# Check if API is running
curl -s http://localhost:8001/ > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ API is ready"
else
    echo "⚠️ API might not be ready yet (will retry)"
fi
echo ""

# Start dashboard
echo "🎨 Starting SolidJS Dashboard (port 3000)..."
cd dashboard
npm run dev &
DASHBOARD_PID=$!
echo "✅ Dashboard started (PID: $DASHBOARD_PID)"
echo ""

echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ SYSTEM RUNNING"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Access points:"
echo "  🎨 Dashboard:  http://localhost:3000"
echo "  📊 API:        http://localhost:8001"
echo "  📖 API Docs:   http://localhost:8001/docs"
echo ""
echo "Processes:"
echo "  API PID: $API_PID"
echo "  Dashboard PID: $DASHBOARD_PID"
echo ""
echo "To stop:"
echo "  kill $API_PID $DASHBOARD_PID"
echo "  Or press Ctrl+C"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Save PIDs for cleanup
echo "$API_PID" > .api_pid
echo "$DASHBOARD_PID" > .dashboard_pid

# Wait for user to stop
wait

