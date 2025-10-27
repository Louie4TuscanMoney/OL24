#!/bin/bash
# Quick status check for training progress

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/Action"

echo "════════════════════════════════════════════════════════════════"
echo "⚡ TRAINING STATUS CHECK"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if LSTM is running
if ps aux | grep -E "🔥_5_TRAIN_LSTM.py" | grep -v grep > /dev/null; then
    echo "🟢 LSTM: TRAINING..."
    
    # Try to find any output files
    if [ -f "model_lstm.pkl" ]; then
        echo "   ✅ LSTM model file exists"
    else
        echo "   ⏳ LSTM model not saved yet"
    fi
else
    echo "⚪ LSTM: Not running"
    
    # Check if it completed
    if [ -f "model_lstm.pkl" ]; then
        echo "   ✅ LSTM: COMPLETE"
    else
        echo "   ❌ LSTM: Not started or failed"
    fi
fi

echo ""
echo "📊 COMPLETED MODELS:"
ls -lh model_*.pkl 2>/dev/null | awk '{print "   "$9" ("$5")"}'

echo ""
echo "🎯 NEXT STEP:"
if [ -f "model_lstm.pkl" ]; then
    echo "   python3 🔥_6_FINAL_VALIDATION.py"
else
    echo "   Wait for LSTM to complete..."
fi

echo "════════════════════════════════════════════════════════════════"

