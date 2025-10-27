#!/bin/bash
################################################################################
# ⚡ RUN ALL RAPID MULTIMODAL IMPLEMENTATION
# Execute all 5 phases in sequence
# Fail forward approach
################################################################################

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/Action"

echo "="
echo "⚡ RAPID MULTIMODAL IMPLEMENTATION - FULL SEQUENCE"
echo "="
echo ""
echo "Philosophy: Fail Forward (Edward Weinhaus)"
echo "Timeline: 4-6 hours"
echo "Risk: High"
echo "Expected outcome: V1 multimodal system or fast failure"
echo ""
echo "="

# Wait for extraction to complete
echo ""
echo "[0/5] Waiting for extraction to complete..."

while true; do
    if [ -f "ULTRA_OPTIMIZED_PATTERNS_2021_2025.pkl" ]; then
        echo "✅ Extraction complete!"
        break
    fi
    
    # Check progress
    if [ -f "stealth_checkpoint.pkl" ]; then
        PROGRESS=$(python3 -c "
import pickle
with open('stealth_checkpoint.pkl', 'rb') as f:
    data = pickle.load(f)
    total = 6913
    done = len(data['patterns'])
    print(f'{done}/{total} ({done/total*100:.1f}%)')
" 2>/dev/null)
        echo "  Extraction: $PROGRESS"
    fi
    
    sleep 60
done

# Phase 1: Collect team stats
echo ""
echo "[1/5] Collecting team stats..."
python3 ⚡_1_collect_team_stats.py

if [ $? -ne 0 ]; then
    echo "❌ Phase 1 failed"
    exit 1
fi

echo "✅ Phase 1 complete"

# Phase 2: Merge features
echo ""
echo "[2/5] Merging team features..."
python3 ⚡_2_merge_features.py

if [ $? -ne 0 ]; then
    echo "❌ Phase 2 failed"
    exit 1
fi

echo "✅ Phase 2 complete"

# Phase 3: Add player features
echo ""
echo "[3/5] Adding player features..."
python3 ⚡_3_add_players_simple.py

if [ $? -ne 0 ]; then
    echo "❌ Phase 3 failed"
    exit 1
fi

echo "✅ Phase 3 complete"

# Phase 4: Train XGBoost
echo ""
echo "[4/5] Training XGBoost..."
python3 ⚡_4_train_xgboost_rapid.py

if [ $? -ne 0 ]; then
    echo "❌ Phase 4 failed"
    exit 1
fi

echo "✅ Phase 4 complete"

# Phase 5: Update game engine
echo ""
echo "[5/6] Updating prediction system..."
python3 ⚡_5_update_game_engine.py

if [ $? -ne 0 ]; then
    echo "❌ Phase 5 failed"
    exit 1
fi

echo "✅ Phase 5 complete"

# Phase 6: TEST ON 2025 AND DECIDE
echo ""
echo "[6/6] Testing on 2025 holdout and making launch decision..."
python3 📊_TEST_2025_AND_DECIDE.py

if [ $? -ne 0 ]; then
    echo "❌ Phase 6 failed"
    exit 1
fi

echo "✅ Phase 6 complete"

# Check decision
DECISION=$(python3 -c "import pickle; d=pickle.load(open('LAUNCH_DECISION.pkl','rb')); print(d['decision'])")

echo ""
echo "="
echo "LAUNCH DECISION: $DECISION"
echo "="

# Summary
echo ""
echo "="
echo "✅ RAPID MULTIMODAL IMPLEMENTATION COMPLETE"
echo "="
echo ""
echo "📊 What was built:"
echo "   ✅ Team feature collection"
echo "   ✅ Feature merging pipeline"
echo "   ✅ Simplified player features"
echo "   ✅ XGBoost model trained"
echo "   ✅ Ensemble prediction system"
echo ""
echo "📈 Features:"
echo "   Before: 57 (PBP only)"
echo "   After: 74 (PBP + Team + Player)"
echo ""
echo "🤖 Models:"
echo "   Before: Dejavu only"
echo "   After: Dejavu + XGBoost ensemble"
echo ""
echo "🧪 Next Steps:"
echo "   1. Test: python3 enhanced_prediction_system.py"
echo "   2. Validate on 2025 holdout"
echo "   3. Compare to baseline"
echo "   4. Launch Monday if better"
echo ""
echo "⚡ Fail Forward: Ship and learn"
echo "="

