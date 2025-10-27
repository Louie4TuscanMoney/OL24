#!/bin/bash
# Quick script to check extraction progress anytime

echo "=================================="
echo "📊 EXTRACTION PROGRESS CHECK"
echo "=================================="

# Check if extraction is running
if pgrep -f "STEALTH_EXTRACTION.py" > /dev/null || pgrep -f "ULTRA_OPTIMIZED_EXTRACTION.py" > /dev/null; then
    echo "✅ Extraction is RUNNING (stealth mode)"
    echo ""
else
    echo "❌ Extraction is NOT running"
    echo ""
fi

# Check checkpoint file
if [ -f "stealth_checkpoint.pkl" ]; then
    echo "📂 Checkpoint file exists"
    
    # Get file size and modification time
    SIZE=$(ls -lh stealth_checkpoint.pkl 2>/dev/null | awk '{print $5}')
    MODIFIED=$(ls -l stealth_checkpoint.pkl 2>/dev/null | awk '{print $6, $7, $8}')
    
    echo "   Size: $SIZE"
    echo "   Last updated: $MODIFIED"
    echo ""
    
    # Try to read progress from Python
    python3 << 'EOF'
import pickle
from pathlib import Path

checkpoint = Path('stealth_checkpoint.pkl')
if checkpoint.exists():
    with open(checkpoint, 'rb') as f:
        data = pickle.load(f)
        processed = len(data['patterns'])
        total = 6914  # Total games to process
        pct = processed / total * 100
        
        print(f"   Games processed: {processed:,} / {total:,} ({pct:.1f}%)")
        
        # Quality breakdown
        grades = {'A': 0, 'B': 0, 'C': 0}
        for pattern in data['patterns']:
            grade = pattern.get('quality_metrics', {}).get('quality_grade', 'C')
            grades[grade] = grades.get(grade, 0) + 1
        
        print(f"   Quality A: {grades['A']:,}")
        print(f"   Quality B: {grades['B']:,}")
        print(f"   Quality C: {grades['C']:,}")
        
        # Estimate time remaining
        if processed > 50:
            avg_time_per_game = 0.7  # seconds
            remaining = total - processed
            remaining_mins = (remaining * avg_time_per_game) / 60
            print(f"   Estimated remaining: {remaining_mins:.0f} minutes")
EOF
else
    echo "📂 No checkpoint file yet (extraction just started)"
fi

echo ""
echo "💡 To see LIVE progress:"
echo "   The extraction script shows updates every 10 games"
echo ""
echo "🔄 To check again, run:"
echo "   bash 📊_CHECK_PROGRESS.sh"

