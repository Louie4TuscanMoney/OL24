#!/bin/bash

# STOP AUTONOMOUS TRADING SYSTEM

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🛑 STOPPING AUTONOMOUS TRADING SYSTEM"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

cd "5. Live System" || exit 1

# Read PIDs
if [ -f ".daemon_pid" ]; then
    DAEMON_PID=$(cat .daemon_pid)
    echo "🤖 Stopping Daemon (PID: $DAEMON_PID)..."
    kill $DAEMON_PID 2>/dev/null
    echo "✅ Daemon stopped"
else
    echo "⚠️ Daemon PID file not found"
    pkill -f "autonomous_trading_daemon.py"
fi

if [ -f ".api_pid" ]; then
    API_PID=$(cat .api_pid)
    echo "📊 Stopping API (PID: $API_PID)..."
    kill $API_PID 2>/dev/null
    echo "✅ API stopped"
else
    echo "⚠️ API PID file not found"
    pkill -f "trading_dashboard_api.py"
fi

# Cleanup
rm -f .daemon_pid .api_pid .pids 2>/dev/null

sleep 2

# Verify stopped
if pgrep -f "autonomous_trading_daemon.py" > /dev/null; then
    echo "⚠️ Daemon still running, force killing..."
    pkill -9 -f "autonomous_trading_daemon.py"
fi

if pgrep -f "trading_dashboard_api.py" > /dev/null; then
    echo "⚠️ API still running, force killing..."
    pkill -9 -f "trading_dashboard_api.py"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ AUTONOMOUS SYSTEM STOPPED"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

