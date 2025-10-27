#!/bin/bash
# Check optimization progress across all trials

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/Action"

echo "=============================================================================="
echo "📊 OPTIMIZATION PROGRESS CHECK"
echo "=============================================================================="
echo ""

# Check which script is running
if ps aux | grep "🔥_2_BAYESIAN_HYPEROPT.py" | grep -v grep > /dev/null; then
    echo "🟢 RUNNING: 100-trial optimization (quick version)"
    echo ""
    
    # Parse log for progress
    if [ -f hyperopt.log ]; then
        echo "Latest progress from hyperopt.log:"
        tail -5 hyperopt.log | grep -E "(Trial|Best trial)"
        echo ""
        
        # Estimate completion
        trials_done=$(tail -100 hyperopt.log | grep -c "Best trial")
        echo "Trials completed: ~$trials_done/100"
        echo "Estimated remaining: ~$((100 - trials_done)) trials"
        echo ""
    fi
    
elif ps aux | grep "🎓_STANFORD_5000_TRIAL_HYPEROPT.py" | grep -v grep > /dev/null; then
    echo "🟢 RUNNING: 5000-trial Stanford optimization (research-grade)"
    echo ""
    
    # Check checkpoint files
    for checkpoint in xgboost_checkpoint.pkl lightgbm_checkpoint.pkl extratrees_checkpoint.pkl randomforest_checkpoint.pkl; do
        if [ -f "$checkpoint" ]; then
            echo "Checkpoint: $checkpoint"
            python3 << EOF
import pickle
try:
    with open('$checkpoint', 'rb') as f:
        cp = pickle.load(f)
    print(f"  Model: {cp['study_name']}")
    print(f"  Trials: {cp['n_trials']}/5000 ({cp['n_trials']/50:.0f}%)")
    print(f"  Best MAE: {cp['best_value']:.4f}")
    print(f"  Elapsed: {cp['elapsed_time']/60:.0f} min")
    print()
except:
    print("  (No data yet)")
    print()
EOF
        fi
    done
    
else
    echo "⏸️  NO OPTIMIZATION RUNNING"
    echo ""
    echo "To start quick optimization (100 trials, ~45 min):"
    echo "  python3 🔥_2_BAYESIAN_HYPEROPT.py"
    echo ""
    echo "To start Stanford optimization (5000 trials, ~40 hours):"
    echo "  nohup python3 🎓_STANFORD_5000_TRIAL_HYPEROPT.py > stanford_hyperopt.log 2>&1 &"
    echo ""
fi

echo "=============================================================================="
echo ""
echo "Current status files:"
ls -lh *checkpoint.pkl 2>/dev/null || echo "  No checkpoints yet"
ls -lh *_trials.db 2>/dev/null || echo "  No SQLite databases yet"
echo ""

echo "To watch live:"
echo "  tail -f hyperopt.log                    (100-trial version)"
echo "  tail -f stanford_hyperopt.log           (5000-trial version)"
echo ""

