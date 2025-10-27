#!/bin/bash
# 🔥 RUN ALL OPTIMIZATIONS
# Complete pipeline to get MAE from 8.22 → 4-5

echo "=============================================================================="
echo "🔥 AGGRESSIVE OPTIMIZATION PIPELINE - GOING FOR CHAMPIONSHIP"
echo "=============================================================================="
echo ""
echo "Target: 4-5 MAE (not settling for 8.22)"
echo "Timeline: 3-5 hours total"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to start..."
sleep 5
echo ""

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/Action"

# Phase 1: Feature engineering (15-20 min)
echo "=============================================================================="
echo "[1/5] ADVANCED FEATURE ENGINEERING"
echo "=============================================================================="
echo "Extracting 50+ features (lag, spectral, momentum, autocorr, advanced stats)"
echo ""
python3 🔥_1_EXTRACT_ALL_FEATURES.py

if [ $? -ne 0 ]; then
    echo "❌ Feature engineering failed"
    exit 1
fi
echo ""

# Phase 2: Hyperparameter optimization (60-90 min)
echo "=============================================================================="
echo "[2/5] BAYESIAN HYPERPARAMETER OPTIMIZATION"
echo "=============================================================================="
echo "Optuna: 100 trials for XGBoost + 100 for ExtraTrees"
echo "This will take 60-90 minutes..."
echo ""
python3 🔥_2_BAYESIAN_HYPEROPT.py

if [ $? -ne 0 ]; then
    echo "❌ Hyperparameter optimization failed"
    exit 1
fi
echo ""

# Phase 3: Train optimized ensemble (15-25 min)
echo "=============================================================================="
echo "[3/5] TRAINING OPTIMIZED ENSEMBLE"
echo "=============================================================================="
echo "Training 4 models with optimal hyperparameters"
echo ""
python3 🔥_3_TRAIN_OPTIMIZED_ENSEMBLE.py

if [ $? -ne 0 ]; then
    echo "❌ Ensemble training failed"
    exit 1
fi
echo ""

# Phase 4: Stack ensemble (10-15 min)
echo "=============================================================================="
echo "[4/5] BUILDING STACKED ENSEMBLE"
echo "=============================================================================="
echo "Meta-learner combining all models"
echo ""
python3 🔥_4_STACK_ENSEMBLE.py

if [ $? -ne 0 ]; then
    echo "❌ Stacking failed"
    exit 1
fi
echo ""

# Phase 5: Add LSTM (20-30 min)
echo "=============================================================================="
echo "[5/5] TRAINING LSTM (DEEP LEARNING)"
echo "=============================================================================="
echo "PyTorch LSTM for temporal patterns"
echo ""
python3 🔥_5_TRAIN_LSTM.py

if [ $? -ne 0 ]; then
    echo "❌ LSTM training failed - but we have ensemble"
fi
echo ""

# Final summary
echo "=============================================================================="
echo "🏆 OPTIMIZATION PIPELINE COMPLETE"
echo "=============================================================================="
echo ""
echo "Results:"
python3 << 'EOF'
import pickle

try:
    with open('FINAL_RESULTS.pkl', 'rb') as f:
        results = pickle.load(f)
    
    print(f"  Baseline (basic XGBoost): 8.22 MAE")
    print(f"  Optimized ensemble:       {results['super_ensemble_mae']:.3f} MAE")
    print(f"  Improvement:              {results['improvement_pct']:.1f}%")
    print()
    
    if results['super_ensemble_mae'] < 5.0:
        print("🏆🏆🏆 CHAMPIONSHIP LEVEL! 🏆🏆🏆")
        print("MAE < 5.0 - READY TO DOMINATE")
    elif results['super_ensemble_mae'] < 6.0:
        print("🏆 EXCELLENT! MAE < 6.0")
        print("Ready for confident launch")
    elif results['super_ensemble_mae'] < 7.0:
        print("✅ GOOD! MAE < 7.0")
        print("Ready for cautious launch")
    else:
        print("⚠️  MAE still above 7.0 - may need more work")

except FileNotFoundError:
    with open('ENSEMBLE_RESULTS.pkl', 'rb') as f:
        results = pickle.load(f)
    print(f"  Best ensemble MAE: {results['best_mae']:.3f}")
    print("  (LSTM not run - see results above)")

EOF

echo ""
echo "All models saved in Action/ directory"
echo ""
echo "Next: Run final validation on 2025 preseason data"
echo "Command: python3 🔥_6_FINAL_VALIDATION.py"
echo ""
echo "=============================================================================="

