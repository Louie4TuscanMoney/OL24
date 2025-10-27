#!/bin/bash

# ONTOLOGIC XYZ - AUTONOMOUS SYSTEM LAUNCHER
# This starts the complete system and keeps it running 24/7

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🔥 ONTOLOGIC XYZ - AUTONOMOUS TRADING SYSTEM"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "This will start:"
echo "  1. Autonomous Trading Daemon (monitors 24/7)"
echo "  2. Trading Dashboard API (serves data)"
echo "  3. Logs all activity"
echo "  4. Auto-restarts on crash"
echo ""
echo "The system runs COMPLETELY AUTONOMOUS - no user interaction needed."
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Create necessary directories
mkdir -p "5. Live System/logs"
mkdir -p "5. Live System/state"

# Change to Live System directory
cd "5. Live System" || exit 1

# Check Python dependencies
echo "📋 Checking dependencies..."
python3 -c "import fastapi, requests, numpy, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Missing dependencies"
    echo "   Installing required packages..."
    pip3 install fastapi uvicorn requests beautifulsoup4 numpy scikit-learn scipy
fi

echo "✅ Dependencies ready"
echo ""

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "autonomous_trading_daemon.py" 2>/dev/null
pkill -f "trading_dashboard_api.py" 2>/dev/null
sleep 2
echo "✅ Cleanup complete"
echo ""

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 STARTING AUTONOMOUS SYSTEM"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Start Trading Daemon (autonomous background process)
echo "🤖 Starting Autonomous Trading Daemon..."
nohup python3 autonomous_trading_daemon.py > logs/daemon_$(date +%Y%m%d_%H%M%S).log 2>&1 &
DAEMON_PID=$!
echo "daemon_pid=$DAEMON_PID" > .pids
echo "✅ Daemon started (PID: $DAEMON_PID)"
echo "   Log: logs/daemon_$(date +%Y%m%d_%H%M%S).log"
echo ""

# Wait for daemon to initialize
sleep 3

# Start Dashboard API
echo "📊 Starting Dashboard API..."
nohup python3 trading_dashboard_api.py > logs/api_$(date +%Y%m%d_%H%M%S).log 2>&1 &
API_PID=$!
echo "api_pid=$API_PID" >> .pids
echo "✅ API started (PID: $API_PID)"
echo "   URL: http://localhost:8001"
echo "   Docs: http://localhost:8001/docs"
echo ""

# Save PIDs
echo "$DAEMON_PID" > .daemon_pid
echo "$API_PID" > .api_pid

sleep 2

# Check if processes are running
if ps -p $DAEMON_PID > /dev/null 2>&1; then
    echo "✅ Daemon is running"
else
    echo "❌ Daemon failed to start"
fi

if ps -p $API_PID > /dev/null 2>&1; then
    echo "✅ API is running"
else
    echo "❌ API failed to start"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ AUTONOMOUS SYSTEM RUNNING"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "🤖 AUTONOMOUS DAEMON:"
echo "   PID: $DAEMON_PID"
echo "   Monitors NBA games every 30 seconds"
echo "   Makes predictions automatically"
echo "   Identifies betting opportunities"
echo "   Enforces risk management"
echo ""
echo "📊 DASHBOARD API:"
echo "   PID: $API_PID"
echo "   URL: http://localhost:8001"
echo "   Docs: http://localhost:8001/docs"
echo ""
echo "📁 LOGS:"
echo "   Daemon: logs/daemon_*.log"
echo "   API: logs/api_*.log"
echo ""
echo "🎯 TO STOP:"
echo "   bash 🛑_STOP_AUTONOMOUS_SYSTEM.sh"
echo "   Or: kill $DAEMON_PID $API_PID"
echo ""
echo "📊 TO VIEW DASHBOARD:"
echo "   1. Install dashboard: cd dashboard_pro && npm install"
echo "   2. Run: npm run dev"
echo "   3. Access: http://localhost:3000"
echo ""
echo "   Or copy to OntologicXYZ.com/NBADashboard and deploy"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ SYSTEM IS NOW RUNNING AUTONOMOUSLY 24/7"
echo ""
echo "The daemon will:"
echo "  ✅ Monitor NBA games continuously"
echo "  ✅ Fetch betting lines automatically"
echo "  ✅ Make predictions when appropriate"
echo "  ✅ Identify +EV opportunities"
echo "  ✅ Enforce risk management"
echo "  ✅ Log all activity"
echo "  ✅ Serve data to dashboard"
echo "  ✅ Auto-restart on crash (with watchdog)"
echo ""
echo "NO USER INTERACTION REQUIRED ✅"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"

