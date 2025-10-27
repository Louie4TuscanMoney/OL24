#!/bin/bash
################################################################################
# 🛡️ EXTRACTION WATCHDOG
# 
# Monitors extraction process and auto-restarts if it crashes
# Runs forever until extraction completes
# Completely hands-off
################################################################################

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/Action"

LOG="watchdog_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"
}

log "🛡️ Watchdog started"

while true; do
    # Check if extraction is complete
    if [ -f "ULTRA_OPTIMIZED_PATTERNS_2021_2025.pkl" ]; then
        log "✅ Extraction complete! Watchdog exiting."
        exit 0
    fi
    
    # Check if process is running AND progressing
    PROCESS_RUNNING=false
    if pgrep -f "STEALTH_EXTRACTION.py" > /dev/null; then
        PROCESS_RUNNING=true
    fi
    
    # Check if checkpoint is updating
    PROGRESS_STUCK=false
    if [ -f "stealth_checkpoint.pkl" ]; then
        CURRENT_PROGRESS=$(python3 -c "
import pickle
try:
    with open('stealth_checkpoint.pkl', 'rb') as f:
        data = pickle.load(f)
        print(len(data['patterns']))
except:
    print(0)
" 2>/dev/null)
        
        # Compare to last known progress
        if [ -f ".last_progress" ]; then
            LAST_PROGRESS=$(cat .last_progress)
            if [ "$CURRENT_PROGRESS" == "$LAST_PROGRESS" ]; then
                # No progress in 60 seconds
                PROGRESS_STUCK=true
            fi
        fi
        
        # Save current progress
        echo "$CURRENT_PROGRESS" > .last_progress
        
        log "✓ Checkpoint: $CURRENT_PROGRESS/6913"
    fi
    
    # Restart if dead OR stuck
    if [ "$PROCESS_RUNNING" = false ] || [ "$PROGRESS_STUCK" = true ]; then
        if [ "$PROCESS_RUNNING" = false ]; then
            log "⚠️  Process died! Auto-restarting..."
        else
            log "⚠️  Process stuck (no progress)! Force-restarting..."
            pkill -9 -f "STEALTH_EXTRACTION.py"
            sleep 2
        fi
        
        # Restart extraction
        nohup python3 -u ⚡_STEALTH_EXTRACTION.py > stealth_extraction.log 2>&1 &
        
        sleep 5
        
        if pgrep -f "STEALTH_EXTRACTION.py" > /dev/null; then
            log "✅ Restarted successfully"
        else
            log "❌ Restart failed"
        fi
    fi
    
    # Check every 60 seconds
    sleep 60
done

