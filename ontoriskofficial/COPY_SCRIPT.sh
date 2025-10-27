#!/bin/bash

echo "🛡️ COPYING ALL ONTORISK FILES..."
echo ""

echo "📂 Copying core framework components..."
cp -v "4. Risk/ontorisk_phase1_probability_calibration.py" ontoriskofficial/components/
cp -v "4. Risk/ontorisk_phase2_edge_calculator.py" ontoriskofficial/components/ 2>/dev/null || echo "  Phase 2 not found (will document)"
cp -v "4. Risk/ontorisk_phase3_kelly_optimizer.py" ontoriskofficial/components/ 2>/dev/null || echo "  Phase 3 not found (will document)"
cp -v "4. Risk/ontorisk_phase4_risk_management.py" ontoriskofficial/components/
cp -v "4. Risk/ontorisk_phase5_archetype_classifier.py" ontoriskofficial/components/

echo ""
echo "📄 Copying specifications and documentation..."
cp -v "4. Risk/🔥_ONTORISK_COMPLETE_SPECIFICATION.md" ontoriskofficial/framework/ 2>/dev/null || echo "  Complete spec not found"
cp -v "4. Risk/🏆_ONTORISK_FINAL_SUMMARY.md" ontoriskofficial/framework/ 2>/dev/null || echo "  Final summary not found"
cp -v "Feel Folder/MODELSYNERGY.md" ontoriskofficial/analysis/ 2>/dev/null || echo "  Model synergy not found"
cp -v "Feel Folder/MODELSYNERGYSUMMARY.md" ontoriskofficial/analysis/ 2>/dev/null || echo "  Model synergy summary not found"

echo ""
echo "✅ CORE FILES COPIED!"
echo "📝 Creating comprehensive documentation..."

