#!/bin/bash
# 🎯 AUTO-QUEUE: Run optimizations sequentially
# 1. Quick 100 trials (45 min) - get results tonight
# 2. Stanford 5000 trials (30-40 hours) - runs overnight automatically
# 3. Forward-feeds best params continuously

echo "=============================================================================="
echo "🎯 INTELLIGENT OPTIMIZATION QUEUE"
echo "=============================================================================="
echo ""
echo "Strategy:"
echo "  1. Quick 100 trials NOW (done by ~4:45 PM)"
echo "  2. Auto-start Stanford 5000 trials (runs overnight)"
echo "  3. Forward-feed best params every 100 trials"
echo ""
echo "Optimizing for: LOW DELTA + HIGH +EV (betting edge)"
echo ""
echo "Starting in 3 seconds..."
sleep 3
echo ""

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/Action"

# Run queued optimization
echo "Starting intelligent queue..."
nohup python3 🎯_INTELLIGENT_OPTIMIZATION_QUEUE.py > intelligent_queue.log 2>&1 &
QUEUE_PID=$!

echo "✅ Optimization queue started (PID: $QUEUE_PID)"
echo ""
echo "This will:"
echo "  Phase 1: Quick 100 trials (~45 min) - completing now"
echo "  Phase 2: Auto-start Stanford 5000 trials after Phase 1"
echo ""
echo "Monitor progress:"
echo "  tail -f intelligent_queue.log"
echo ""
echo "Check status:"
echo "  bash 📊_CHECK_OPTIMIZATION.sh"
echo ""
echo "Timeline:"
echo "  ~4:45 PM - Phase 1 complete (100 trials, MAE ~9.5-10)"
echo "  ~4:45 PM - Phase 2 auto-starts (5000 trials)"
echo "  Sunday ~6 PM - Phase 2 complete (MAE ~5-6 target)"
echo ""
echo "Checkpoints saved every 100 trials to:"
echo "  - stanford_xgboost_5000_checkpoint.pkl"
echo "  - stanford_lightgbm_5000_checkpoint.pkl"
echo "  - stanford_extratrees_5000_checkpoint.pkl"
echo ""
echo "Current best params continuously updated to:"
echo "  - stanford_xgboost_5000_CURRENT_BEST.pkl"
echo "  - stanford_lightgbm_5000_CURRENT_BEST.pkl"
echo "  - stanford_extratrees_5000_CURRENT_BEST.pkl"
echo ""
echo "=============================================================================="
echo "LET IT RUN! Building a championship system! 🏆"
echo "=============================================================================="

