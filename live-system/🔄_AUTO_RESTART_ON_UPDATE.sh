#!/bin/bash

# AUTO-RESTART BACKEND WHEN CODE CHANGES
# Watches nba_live_scores.py and auto-restarts backend

echo "🔄 Starting auto-restart watcher..."
echo "Monitoring: nba_live_scores.py"
echo "Press Ctrl+C to stop"
echo ""

# Get the file modification time
WATCH_FILE="5. Live System/nba_live_scores.py"
LAST_MODIFIED=$(stat -f %m "$WATCH_FILE" 2>/dev/null || stat -c %Y "$WATCH_FILE" 2>/dev/null)

# Start backend
echo "🚀 Starting backend..."
bash 🚀_START_AUTONOMOUS_SYSTEM.sh &
BACKEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo ""

# Watch for changes
while true; do
    sleep 5
    
    # Check if file was modified
    CURRENT_MODIFIED=$(stat -f %m "$WATCH_FILE" 2>/dev/null || stat -c %Y "$WATCH_FILE" 2>/dev/null)
    
    if [ "$CURRENT_MODIFIED" != "$LAST_MODIFIED" ]; then
        echo ""
        echo "🔄 Code updated! Auto-restarting backend..."
        
        # Kill old backend
        kill $BACKEND_PID 2>/dev/null
        pkill -f trading_dashboard_api 2>/dev/null
        sleep 2
        
        # Restart
        bash 🚀_START_AUTONOMOUS_SYSTEM.sh &
        BACKEND_PID=$!
        
        LAST_MODIFIED=$CURRENT_MODIFIED
        echo "✅ Backend restarted! PID: $BACKEND_PID"
        echo ""
    fi
done

