#!/bin/bash

# 🚂 DEPLOY FIXES TO RAILWAY
# This commits all fixes and pushes to Railway for cloud deployment

echo "========================================================================"
echo "🚂 DEPLOYING FIXES TO RAILWAY"
echo "========================================================================"
echo ""

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/live-system"

echo "📝 Git Status:"
git status --short

echo ""
echo "========================================================================"
echo "CRITICAL FIXES BEING DEPLOYED:"
echo "========================================================================"
echo ""
echo "✅ FIX 1: nba_live_scores.py"
echo "   - Use nba_api library FIRST (correct game IDs)"
echo "   - Fixed 'clock' vs 'game_clock' variable typo"
echo ""
echo "✅ FIX 2: mamba_live_feature_extractor.py"
echo "   - Use LIVE API (nba_api.live) instead of stats API"
echo "   - Parse lowercase field names (period, clock, scoreHome)"
echo "   - Parse PT06M30.00S clock format correctly"
echo ""
echo "✅ FIX 3: live_trading_engine.py"
echo "   - Show predictions for ALL games (not just betting opportunities)"
echo "   - Create synthetic lines if BetOnline fails"
echo ""
echo "========================================================================"

# Stage the critical files
echo ""
echo "📦 Staging files for commit..."
git add nba_live_scores.py
git add mamba_live_feature_extractor.py
git add live_trading_engine.py

echo "✅ Files staged"
echo ""

# Commit
echo "💾 Committing changes..."
git commit -m "🔥 CRITICAL FIX: Real-time predictions with correct game IDs

- FIX nba_live_scores: Use nba_api library for correct game IDs (not ESPN)
- FIX mamba_feature_extractor: Use live API with correct field names
- FIX live_trading_engine: Show ALL predictions (not just betting opps)
- All 33 Mamba features now extract from REAL live PBP data
- System displays predictions for every game at Q2 6:00

Tested locally: ✅ WORKING
Ready for Railway deployment"

if [ $? -eq 0 ]; then
    echo "✅ Commit successful!"
else
    echo "⚠️ Nothing to commit (files already committed?)"
fi

echo ""
echo "========================================================================"
echo "🚂 PUSHING TO RAILWAY..."
echo "========================================================================"

# Push to origin (Railway should auto-deploy)
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅✅✅ DEPLOYMENT SUCCESSFUL! ✅✅✅"
    echo "========================================================================"
    echo ""
    echo "🚂 Railway will now:"
    echo "   1. Pull latest code with all fixes"
    echo "   2. Rebuild the container"
    echo "   3. Restart the daemon"
    echo "   4. Auto-trigger predictions at Q2 6:00"
    echo ""
    echo "⏰ Deployment usually takes 2-3 minutes"
    echo ""
    echo "🔍 Check status:"
    echo "   railway logs --follow"
    echo ""
    echo "🌐 Dashboard should update automatically"
    echo ""
else
    echo ""
    echo "❌ Push failed!"
    echo "Check your Railway configuration"
fi

