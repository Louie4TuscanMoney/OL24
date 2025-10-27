#!/bin/bash
# Check status of queued optimizations with detailed progress

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/Action"

echo "=============================================================================="
echo "📊 OPTIMIZATION QUEUE STATUS"
echo "=============================================================================="
echo ""

# Check if queue is running
if ps aux | grep "🎯_INTELLIGENT_OPTIMIZATION_QUEUE.py" | grep -v grep > /dev/null; then
    echo "🟢 QUEUE IS RUNNING"
    echo ""
    
    # Check which phase
    if [ -f "QUICK_100_RESULTS.pkl" ]; then
        echo "Phase 1: ✅ COMPLETE (100 trials)"
        python3 << 'EOF'
import pickle
with open('QUICK_100_RESULTS.pkl', 'rb') as f:
    results = pickle.load(f)
print(f"  Best MAE: {results['xgboost']['best_mae']:.3f}")
print(f"  Best Delta: {results['xgboost']['best_delta']:.3f}")
EOF
        echo ""
        echo "Phase 2: 🔄 STANFORD 5000 TRIALS RUNNING"
        echo ""
    else
        echo "Phase 1: 🔄 RUNNING (100 trials)"
        tail -5 intelligent_queue.log | grep -E "(Trial|Best)" | tail -3
        echo ""
        echo "Phase 2: ⏳ QUEUED (will auto-start after Phase 1)"
        echo ""
    fi
    
    # Check Stanford checkpoints
    for checkpoint in stanford_*_checkpoint.pkl; do
        if [ -f "$checkpoint" ]; then
            echo "Checkpoint: $checkpoint"
            python3 << EOF
import pickle
try:
    with open('$checkpoint', 'rb') as f:
        cp = pickle.load(f)
    print(f"  Progress: {cp['n_trials']}/5000 ({cp['n_trials']/50:.0f}%)")
    print(f"  Best MAE: {cp['best_mae']:.4f} (Delta: {cp['best_delta']:.4f})")
    print(f"  Elapsed: {cp['elapsed_min']:.0f} min")
    print(f"  Est. remaining: {cp['est_remaining_min']:.0f} min ({cp['est_remaining_min']/60:.1f} hours)")
    print()
except:
    pass
EOF
        fi
    done
    
else
    echo "⏸️  QUEUE NOT RUNNING"
    echo ""
    echo "To start queue:"
    echo "  bash 🎯_AUTO_QUEUE_OPTIMIZATIONS.sh"
    echo ""
    
    # Check if quick results exist
    if [ -f "QUICK_100_RESULTS.pkl" ]; then
        echo "✅ Quick 100-trial results available"
        python3 << 'EOF'
import pickle
with open('QUICK_100_RESULTS.pkl', 'rb') as f:
    results = pickle.load(f)
print(f"  MAE: {results['xgboost']['best_mae']:.3f}")
EOF
        echo ""
    fi
    
    # Check if Stanford results exist
    if [ -f "STANFORD_FINAL_RESULTS.pkl" ]; then
        echo "✅ Stanford 5000-trial results available"
        python3 << 'EOF'
import pickle
with open('STANFORD_FINAL_RESULTS.pkl', 'rb') as f:
    results = pickle.load(f)
print(f"  Best model: {results['best_model']}")
print(f"  Best MAE: {results['best_mae']:.4f}")
print(f"  Best Delta: {results['best_delta']:.4f}")
EOF
        echo ""
    fi
fi

echo "=============================================================================="
echo ""
echo "Files:"
ls -lh *_checkpoint.pkl 2>/dev/null | awk '{print "  "$9" ("$5")"}'
ls -lh *_CURRENT_BEST.pkl 2>/dev/null | awk '{print "  "$9" ("$5")"}'
ls -lh stanford_*.db 2>/dev/null | awk '{print "  "$9" ("$5")"}'
echo ""

echo "Monitor live:"
echo "  tail -f intelligent_queue.log"
echo ""

