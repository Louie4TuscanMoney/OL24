#!/bin/bash
################################################################################
# 📊 DETAILED PROGRESS MONITOR
# 
# Shows comprehensive status of entire pipeline
# Run anytime to see where things are
################################################################################

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/Action"

clear

cat << 'EOF'
================================================================================
📊 COMPREHENSIVE PROGRESS MONITOR
================================================================================
EOF

echo "Current Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

################################################################################
# CHECK 1: EXTRACTION STATUS
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  EXTRACTION STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if extraction is running
if pgrep -f "ULTRA_OPTIMIZED_EXTRACTION" > /dev/null; then
    echo "   Status: 🟢 RUNNING"
    
    # Get progress
    if [ -f "ultra_optimized_checkpoint.pkl" ]; then
        python3 << 'PYEOF'
import pickle
try:
    with open('ultra_optimized_checkpoint.pkl', 'rb') as f:
        data = pickle.load(f)
        processed = len(data['patterns'])
        total = 6913
        pct = processed / total * 100
        
        # Calculate ETA (using realistic 0.7 sec per game)
        remaining = total - processed
        secs_per_game = 0.7
        eta_secs = remaining * secs_per_game
        eta_mins = eta_secs / 60
        eta_hours = eta_mins / 60
        
        print(f"   Progress: {processed:,} / {total:,} games ({pct:.1f}%)")
        if eta_hours >= 1:
            print(f"   Estimated remaining: {eta_hours:.1f} hours ({eta_mins:.0f} min)")
        else:
            print(f"   Estimated remaining: {eta_mins:.0f} minutes")
        
        # Progress bar
        bar_length = 50
        filled = int(bar_length * pct / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"   [{bar}]")
        
except Exception as e:
    print(f"   Error reading checkpoint: {e}")
PYEOF
    fi
else
    # Not running - check if complete
    if [ -f "ULTRA_OPTIMIZED_PATTERNS_2021_2025.pkl" ]; then
        echo "   Status: ✅ COMPLETE"
        
        python3 << 'PYEOF'
import pickle
try:
    with open('ULTRA_OPTIMIZED_PATTERNS_2021_2025.pkl', 'rb') as f:
        patterns = pickle.load(f)
        print(f"   Total games extracted: {len(patterns):,}")
        
        # Quality check
        quality_grades = [p.get('quality_metrics', {}).get('quality_grade', 'C') for p in patterns]
        from collections import Counter
        grade_counts = Counter(quality_grades)
        
        print(f"   Quality distribution:")
        for grade in ['A', 'B', 'C']:
            count = grade_counts.get(grade, 0)
            pct = count / len(patterns) * 100 if len(patterns) > 0 else 0
            print(f"      Grade {grade}: {count:,} ({pct:.1f}%)")
except:
    print("   Error reading extraction file")
PYEOF
    else
        echo "   Status: ⏳ NOT STARTED or FAILED"
        echo "   File not found: ULTRA_OPTIMIZED_PATTERNS_2021_2025.pkl"
    fi
fi

echo ""

################################################################################
# CHECK 2: MERGE STATUS
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  DATA MERGE STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "merge_metadata.pkl" ]; then
    echo "   Status: ✅ COMPLETE"
    
    python3 << 'PYEOF'
import pickle
try:
    with open('merge_metadata.pkl', 'rb') as f:
        meta = pickle.load(f)
        print(f"   New games: {meta['new_games']:,}")
        print(f"   Existing games: {meta['old_games']:,}")
        print(f"   Total dataset: {meta['total_games']:,}")
        print(f"   Merge date: {meta['merge_date'][:19]}")
except:
    print("   Error reading merge metadata")
PYEOF
else
    echo "   Status: ⏳ PENDING"
fi

echo ""

################################################################################
# CHECK 3: MODEL TRAINING STATUS
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  MODEL TRAINING STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "1. ML/1. Dejavu Deployment/dejavu_retrained_2025.pkl" ]; then
    echo "   Status: ✅ COMPLETE"
    
    python3 << 'PYEOF'
import pickle
import sys
sys.path.insert(0, '1. ML/1. Dejavu Deployment')
try:
    from dejavu_model import DejavuForecaster
    model = DejavuForecaster.load('1. ML/1. Dejavu Deployment/dejavu_retrained_2025.pkl')
    print(f"   Database size: {len(model.database):,} patterns")
    print(f"   K-nearest neighbors: {model.k}")
except Exception as e:
    print(f"   Error loading model: {e}")
PYEOF
else
    echo "   Status: ⏳ PENDING"
fi

echo ""

################################################################################
# CHECK 4: EVALUATION STATUS
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣  EVALUATION STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "evaluation_metrics.pkl" ]; then
    echo "   Status: ✅ COMPLETE"
    
    python3 << 'PYEOF'
import pickle
try:
    with open('evaluation_metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
        
        mae = metrics['mae']
        rmse = metrics['rmse']
        r2 = metrics['r2']
        
        print(f"   MAE: {mae:.2f} points")
        print(f"   RMSE: {rmse:.2f} points")
        print(f"   R²: {r2:.3f}")
        print(f"   Holdout games: {metrics['n_holdout']}")
        
        # Compare to old
        old_mae = 10.75
        improvement = ((old_mae - mae) / old_mae) * 100
        print(f"")
        print(f"   Comparison:")
        print(f"      Old MAE: {old_mae:.2f}")
        print(f"      New MAE: {mae:.2f}")
        print(f"      Improvement: {improvement:.1f}%")
        
        if mae < 7.0:
            print(f"      🎯 TARGET ACHIEVED!")
        else:
            print(f"      ⚠️  Target: <7.0 (need tuning)")
except Exception as e:
    print(f"   Error: {e}")
PYEOF
else
    echo "   Status: ⏳ PENDING"
fi

echo ""

################################################################################
# CHECK 5: OVERALL PIPELINE STATUS
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5️⃣  OVERALL PIPELINE STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "PIPELINE_COMPLETE.marker" ]; then
    echo "   Status: ✅ FULLY COMPLETE"
    echo ""
    echo "   Next steps:"
    echo "      1. Read EXECUTIVE_SUMMARY.txt"
    echo "      2. Review evaluation metrics"
    echo "      3. Test prediction on live game"
    echo "      4. Launch Monday 4 PM PST"
else
    echo "   Status: 🔄 IN PROGRESS"
    echo ""
    echo "   To check again, run:"
    echo "      bash 📊_DETAILED_PROGRESS.sh"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check for executive summary
if [ -f "EXECUTIVE_SUMMARY.txt" ]; then
    echo "📋 Executive summary available:"
    echo "   cat EXECUTIVE_SUMMARY.txt"
    echo ""
fi

# Check for logs
LATEST_LOG=$(ls -t pipeline_execution_*.log 2>/dev/null | head -1)
if [ ! -z "$LATEST_LOG" ]; then
    echo "📝 Latest log file:"
    echo "   tail -f $LATEST_LOG"
    echo ""
fi

echo "Shabbat Shalom 🕯️"
echo ""

