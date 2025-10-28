#!/bin/bash

# 🚀 START FULL AUTOMATED SYSTEM
# This launches everything you need:
# 1. Backend API (FastAPI + WebSocket)
# 2. Autonomous daemon (monitors games, auto-triggers at Q2 6:00)
# 3. Frontend dashboard (React/SolidJS)

echo "========================================================================"
echo "🚀 ONTOLOGIC XYZ - FULL SYSTEM STARTUP"
echo "========================================================================"
echo ""
echo "Starting components:"
echo "  1. Trading Dashboard API (FastAPI + WebSocket)"
echo "  2. Autonomous Trading Daemon (Q2 6:00 auto-trigger)"
echo "  3. Frontend Dashboard (React)"
echo ""
echo "========================================================================"

# Create logs directory
mkdir -p logs

# Kill any existing processes on our ports
echo "🧹 Cleaning up existing processes..."
pkill -f "trading_dashboard_api.py" 2>/dev/null
pkill -f "autonomous_trading_daemon.py" 2>/dev/null
pkill -f "vite" 2>/dev/null
sleep 2

# ============================================================================
# STEP 1: Start Backend API (port 8000)
# ============================================================================
echo ""
echo "📡 Starting Trading Dashboard API (port 8000)..."
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/live-system"
nohup python3 trading_dashboard_api.py > logs/api_$(date +%Y%m%d_%H%M%S).log 2>&1 &
API_PID=$!
echo "   ✅ API started (PID: $API_PID)"
echo "   📊 Swagger docs: http://localhost:8000/docs"
sleep 3

# ============================================================================
# STEP 2: Start Autonomous Daemon (monitors games)
# ============================================================================
echo ""
echo "🤖 Starting Autonomous Trading Daemon..."
nohup python3 autonomous_trading_daemon.py > logs/daemon_$(date +%Y%m%d_%H%M%S).log 2>&1 &
DAEMON_PID=$!
echo "   ✅ Daemon started (PID: $DAEMON_PID)"
echo "   🎯 Auto-triggers at Q2 6:00"
echo "   📝 Check logs: tail -f logs/daemon_$(date +%Y%m%d).log"
sleep 3

# ============================================================================
# STEP 3: Start Frontend Dashboard (port 5173)
# ============================================================================
echo ""
echo "🎨 Starting Frontend Dashboard (port 5173)..."

# Check which dashboard exists
if [ -d "dashboard_pro" ]; then
    cd dashboard_pro
    DASHBOARD_DIR="dashboard_pro"
elif [ -d "dashboard" ]; then
    cd dashboard
    DASHBOARD_DIR="dashboard"
elif [ -d "../Action/5. Frontend/nba-dashboard" ]; then
    cd "../Action/5. Frontend/nba-dashboard"
    DASHBOARD_DIR="Action/5. Frontend/nba-dashboard"
else
    echo "   ⚠️ No dashboard found, skipping..."
    DASHBOARD_DIR=""
fi

if [ -n "$DASHBOARD_DIR" ]; then
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "   📦 Installing dependencies..."
        npm install --silent
    fi
    
    # Start dev server
    nohup npm run dev > ../logs/dashboard_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    DASHBOARD_PID=$!
    echo "   ✅ Dashboard started (PID: $DASHBOARD_PID)"
    echo "   🌐 Dashboard: http://localhost:5173"
fi

# ============================================================================
# DONE!
# ============================================================================
echo ""
echo "========================================================================"
echo "✅ ALL SYSTEMS RUNNING!"
echo "========================================================================"
echo ""
echo "🎯 What happens now:"
echo "   1. Daemon checks games every 3 seconds"
echo "   2. At Q2 6:00, automatically:"
echo "      - Fetches 18-min play-by-play"
echo "      - Extracts ALL 33 Mamba features"
echo "      - Runs ML prediction"
echo "      - Sends to dashboard via WebSocket"
echo "   3. You just watch the dashboard!"
echo ""
echo "🌐 Access Points:"
echo "   Dashboard:  http://localhost:5173"
echo "   API:        http://localhost:8000"
echo "   API Docs:   http://localhost:8000/docs"
echo ""
echo "📊 Monitor Logs:"
echo "   API:     tail -f logs/api_$(date +%Y%m%d)*.log"
echo "   Daemon:  tail -f logs/daemon_$(date +%Y%m%d).log"
echo ""
echo "🛑 Stop Everything:"
echo "   pkill -f trading_dashboard_api"
echo "   pkill -f autonomous_trading_daemon"
echo "   pkill -f vite"
echo ""
echo "========================================================================"
echo "🚀 SYSTEM READY - NO TOUCHING NEEDED!"
echo "========================================================================"

