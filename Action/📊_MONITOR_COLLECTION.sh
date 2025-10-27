#!/bin/bash

# Monitor data collection progress

echo "════════════════════════════════════════════════════════════════════════════════"
echo "📊 DATA COLLECTION MONITOR"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if collection is running
if pgrep -f "COLLECT_2015_2020" > /dev/null; then
    echo "✅ Collection IS RUNNING in background"
else
    echo "⚠️  Collection NOT running (may have completed or not started)"
fi

echo ""
echo "Checking for checkpoint files..."
echo ""

# Find checkpoint files
CHECKPOINT_DIR="Action"
CHECKPOINTS=$(ls -t $CHECKPOINT_DIR/COLLECTION_CHECKPOINT_*.pkl 2>/dev/null | head -5)

if [ -z "$CHECKPOINTS" ]; then
    echo "  No checkpoints found yet"
    echo "  (Collection may still be in game ID gathering phase)"
else
    echo "  Latest checkpoints:"
    for cp in $CHECKPOINTS; do
        SIZE=$(ls -lh "$cp" | awk '{print $5}')
        TIME=$(ls -l "$cp" | awk '{print $6, $7, $8}')
        echo "    • $(basename $cp): $SIZE (modified: $TIME)"
    done
fi

echo ""
echo "Checking log file..."
echo ""

# Check log
LOG_FILE="Action/collection_2015_2020.log"

if [ -f "$LOG_FILE" ]; then
    echo "  Log file exists: $(ls -lh $LOG_FILE | awk '{print $5}')"
    echo ""
    echo "  Last 10 lines:"
    tail -10 "$LOG_FILE" | sed 's/^/    /'
else
    echo "  No log file yet"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "To check full log: tail -f Action/collection_2015_2020.log"
echo "To check checkpoints: ls -lh Action/COLLECTION_CHECKPOINT_*.pkl"
echo ""
echo "Expected completion: 1-2 hours from start"
echo "Target: 7,400+ new games collected"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"

