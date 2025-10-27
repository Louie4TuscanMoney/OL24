#!/bin/bash
################################################################################
# SMART WATCHDOG - Reusable Process Monitor
# 
# Monitors any Python script and auto-restarts if:
# - Process crashes
# - Process gets stuck (no progress)
# 
# Usage: bash smart_watchdog.sh your_script.py
################################################################################

if [ $# -eq 0 ]; then
    echo "Usage: bash smart_watchdog.sh script_to_monitor.py"
    exit 1
fi

SCRIPT_TO_MONITOR="$1"
SCRIPT_NAME=$(basename "$SCRIPT_TO_MONITOR" .py)
LOG_FILE="watchdog_${SCRIPT_NAME}_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "🛡️ Smart Watchdog started for: $SCRIPT_TO_MONITOR"

# Find checkpoint file (assumes pattern: *checkpoint.pkl)
find_checkpoint() {
    find . -maxdepth 1 -name "*checkpoint.pkl" -type f | head -1
}

while true; do
    # Check if output file exists (completion marker)
    if ls ULTRA_OPTIMIZED_PATTERNS_*.pkl 2>/dev/null | grep -q .; then
        log "✅ Output file detected - Task complete!"
        exit 0
    fi
    
    # Check if process is running
    PROCESS_RUNNING=false
    if pgrep -f "$SCRIPT_TO_MONITOR" > /dev/null; then
        PROCESS_RUNNING=true
    fi
    
    # Check progress via checkpoint
    CHECKPOINT_FILE=$(find_checkpoint)
    PROGRESS_STUCK=false
    
    if [ ! -z "$CHECKPOINT_FILE" ] && [ -f "$CHECKPOINT_FILE" ]; then
        CURRENT_PROGRESS=$(python3 -c "
import pickle
try:
    with open('$CHECKPOINT_FILE', 'rb') as f:
        data = pickle.load(f)
        if 'patterns' in data:
            print(len(data['patterns']))
        elif 'items' in data:
            print(len(data['items']))
        else:
            print(0)
except:
    print(0)
" 2>/dev/null)
        
        # Compare to last
        if [ -f ".${SCRIPT_NAME}_last_progress" ]; then
            LAST_PROGRESS=$(cat ".${SCRIPT_NAME}_last_progress")
            if [ "$CURRENT_PROGRESS" == "$LAST_PROGRESS" ] && [ "$PROCESS_RUNNING" = true ]; then
                PROGRESS_STUCK=true
            fi
        fi
        
        # Save current
        echo "$CURRENT_PROGRESS" > ".${SCRIPT_NAME}_last_progress"
        
        if [ "$PROCESS_RUNNING" = true ]; then
            log "✓ Running - Progress: $CURRENT_PROGRESS"
        fi
    fi
    
    # Restart if needed
    if [ "$PROCESS_RUNNING" = false ] || [ "$PROGRESS_STUCK" = true ]; then
        if [ "$PROCESS_RUNNING" = false ]; then
            log "⚠️  Process died! Auto-restarting..."
        else
            log "⚠️  Process stuck (no progress for 60s)! Force-restarting..."
            pkill -9 -f "$SCRIPT_TO_MONITOR"
            sleep 2
        fi
        
        # Restart
        nohup python3 -u "$SCRIPT_TO_MONITOR" > "${SCRIPT_NAME}_output.log" 2>&1 &
        
        sleep 5
        
        if pgrep -f "$SCRIPT_TO_MONITOR" > /dev/null; then
            log "✅ Restarted successfully"
        else
            log "❌ Restart failed - manual intervention needed"
        fi
    fi
    
    # Check every 60 seconds
    sleep 60
done

