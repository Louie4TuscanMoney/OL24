#!/bin/bash

# AUTONOMOUS WATCHDOG - Keeps the system running 24/7
# This monitors the daemon and API, auto-restarting them if they crash

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🔄 AUTONOMOUS TRADING WATCHDOG"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "This watchdog ensures the trading system NEVER stops."
echo "It monitors processes and auto-restarts them on crash."
echo ""
echo "Press Ctrl+C to stop (not recommended for production)"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Configuration
CHECK_INTERVAL=60  # Check every 60 seconds
MAX_RESTARTS=10    # Max restarts per hour
RESTART_COUNT=0
LAST_RESTART_TIME=0

cd "5. Live System" || exit 1

# Function to start daemon
start_daemon() {
    echo "[$(date +'%H:%M:%S')] 🤖 Starting Daemon..."
    nohup python3 autonomous_trading_daemon.py > logs/daemon_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    DAEMON_PID=$!
    echo "$DAEMON_PID" > .daemon_pid
    echo "[$(date +'%H:%M:%S')] ✅ Daemon started (PID: $DAEMON_PID)"
}

# Function to start API
start_api() {
    echo "[$(date +'%H:%M:%S')] 📊 Starting API..."
    nohup python3 trading_dashboard_api.py > logs/api_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    API_PID=$!
    echo "$API_PID" > .api_pid
    echo "[$(date +'%H:%M:%S')] ✅ API started (PID: $API_PID)"
}

# Initial start
start_daemon
start_api

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ WATCHDOG ACTIVE - Monitoring every ${CHECK_INTERVAL}s"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Monitor loop
while true; do
    sleep $CHECK_INTERVAL
    
    # Check daemon
    if [ -f ".daemon_pid" ]; then
        DAEMON_PID=$(cat .daemon_pid)
        if ! ps -p $DAEMON_PID > /dev/null 2>&1; then
            echo ""
            echo "🚨 [$(date +'%H:%M:%S')] DAEMON CRASHED - Auto-restarting..."
            
            # Rate limit restarts
            CURRENT_TIME=$(date +%s)
            if [ $((CURRENT_TIME - LAST_RESTART_TIME)) -lt 3600 ]; then
                RESTART_COUNT=$((RESTART_COUNT + 1))
            else
                RESTART_COUNT=1
            fi
            LAST_RESTART_TIME=$CURRENT_TIME
            
            if [ $RESTART_COUNT -gt $MAX_RESTARTS ]; then
                echo "🚨 TOO MANY RESTARTS ($RESTART_COUNT in 1 hour)"
                echo "   Stopping watchdog - manual intervention needed"
                exit 1
            fi
            
            start_daemon
            echo "   Restart count: $RESTART_COUNT/$MAX_RESTARTS"
            echo ""
        fi
    else
        echo "⚠️ [$(date +'%H:%M:%S')] Daemon PID file missing - restarting..."
        start_daemon
    fi
    
    # Check API
    if [ -f ".api_pid" ]; then
        API_PID=$(cat .api_pid)
        if ! ps -p $API_PID > /dev/null 2>&1; then
            echo ""
            echo "🚨 [$(date +'%H:%M:%S')] API CRASHED - Auto-restarting..."
            start_api
            echo ""
        fi
    else
        echo "⚠️ [$(date +'%H:%M:%S')] API PID file missing - restarting..."
        start_api
    fi
    
    # Heartbeat
    if [ $(($(date +%s) % 300)) -lt $CHECK_INTERVAL ]; then
        echo "💓 [$(date +'%H:%M:%S')] Watchdog heartbeat - All systems operational"
    fi
done

