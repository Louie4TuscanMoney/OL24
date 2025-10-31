#!/bin/bash
# Deploy API fixes to Railway

echo "🚀 DEPLOYING API FIXES TO RAILWAY"
echo "================================================================================"
echo ""

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research"

echo "1️⃣  Checking git status..."
git status

echo ""
echo "2️⃣  Adding updated API file..."
git add live-system/trading_dashboard_api.py

echo ""
echo "3️⃣  Committing changes..."
git commit -m "Fix API endpoints: use team_season_stats, add advanced player stats (BPM, PER, VORP)" || echo "Nothing to commit (already committed?)"

echo ""
echo "4️⃣  Pushing to Railway (will auto-deploy)..."
git push origin main

echo ""
echo "================================================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "================================================================================"
echo ""
echo "Railway will auto-deploy in ~2-3 minutes."
echo ""
echo "🧪 Test the API after deployment:"
echo ""
echo "   # Test teams endpoint:"
echo "   curl -s https://ol24-production.up.railway.app/api/stats/teams | jq '.teams[0]'"
echo ""
echo "   # Test player endpoint (Luka):"
echo "   curl -s https://ol24-production.up.railway.app/api/stats/player/1629029 | jq '.season_stats'"
echo ""
echo "   # Test standings:"
echo "   curl -s https://ol24-production.up.railway.app/api/stats/standings | jq '.standings.East | length'"
echo ""
echo "================================================================================"
echo "📊 Your frontend can now display:"
echo "   ✅ Team logos + colors + W-L records + PPG"
echo "   ✅ Player headshots + first/last names + positions"
echo "   ✅ Player stats (PPG, RPG, APG, FG%, advanced metrics)"
echo "   ✅ Advanced stats (BPM, PER, VORP, Win Shares, Usage%)"
echo "   ✅ Standings (conference, rank, streak)"
echo "================================================================================"

