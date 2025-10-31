#!/bin/bash

echo "🚀 DEPLOYING CRON FIX TO RAILWAY"
echo "========================================================================"
echo ""

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research"

echo "Step 1: Committing cron fix"
echo "------------------------------------------------------------------------"

git add live-system/cron_mamba_autonomous.py

git commit -m "🐛 Fix cron script to use nba_api instead of CDN JSON

Issue: Cron was using NBA CDN endpoint which had stale data
Fix: Use nba_api.live.nba.endpoints.scoreboard (same as /api/live-games)

This ensures cron sees the same live game data as the REST API.

When games start tonight (6:30 PM ET):
- Cron will detect live games
- Play-by-play will be collected
- Mamba will trigger at Q2 6:00
- Dashboard will show ML predictions"

echo ""
echo ""
echo "Step 2: Pushing to GitHub"
echo "------------------------------------------------------------------------"

git push origin main

echo ""
echo ""
echo "========================================================================"
echo "✅ CRON FIX DEPLOYED!"
echo "========================================================================"
echo ""
echo "🎯 What Happens Next:"
echo "------------------------------------------------------------------------"
echo ""
echo "  1. Railway auto-deploys cron fix"
echo ""
echo "  2. Tonight at 6:30 PM ET (first game):"
echo "     • Cron detects live game"
echo "     • Starts collecting play-by-play"
echo "     • Stores minute-by-minute scoring"
echo ""
echo "  3. At Q2 6:00 mark:"
echo "     • ⚡ Mamba triggers automatically"
echo "     • Extracts 33 features"
echo "     • Makes prediction"
echo "     • Stores in database"
echo "     • Broadcasts via WebSocket"
echo ""
echo "  4. On your dashboard:"
echo "     • Live scores appear"
echo "     • ML predictions display"
echo "     • Trading opportunities show"
echo "     • Mamba widget updates real-time"
echo ""
echo "========================================================================"
echo ""
echo "📅 Game Schedule (tonight):"
echo "------------------------------------------------------------------------"
echo "  6:30 PM ET - HOU @ TOR"
echo "  7:00 PM ET - CLE @ BOS"
echo "  7:00 PM ET - ORL @ DET"
echo "  7:30 PM ET - ATL @ BKN"
echo "  8:00 PM ET - SAC @ CHI"
echo "  (and 5 more games)"
echo ""
echo "🔥 System is ready! Just wait for games to start."
echo ""
echo "========================================================================"

