#!/bin/bash

# 🚂 RAILWAY DEPLOYMENT TEST SCRIPT
# Replace YOUR_RAILWAY_URL with your actual Railway public URL

RAILWAY_URL="YOUR_RAILWAY_URL_HERE"  # e.g., "https://your-app.up.railway.app"

echo "🚂 TESTING RAILWAY DEPLOYMENT"
echo "================================"
echo ""

# Test 1: Health Check
echo "✅ TEST 1: Health Check"
curl -s "${RAILWAY_URL}/" | jq '.'
echo ""
echo ""

# Test 2: Live Games
echo "✅ TEST 2: Live Games"
curl -s "${RAILWAY_URL}/api/live-games" | jq '.'
echo ""
echo ""

# Test 3: BetOnline Odds
echo "✅ TEST 3: BetOnline Odds"
curl -s "${RAILWAY_URL}/api/betonline-odds" | jq '.'
echo ""
echo ""

# Test 4: Trading Opportunities
echo "✅ TEST 4: Trading Opportunities"
curl -s "${RAILWAY_URL}/api/opportunities" | jq '.'
echo ""
echo ""

echo "================================"
echo "🎉 ALL TESTS COMPLETE!"
echo ""
echo "If you see JSON responses above, your backend is LIVE! 🔥"

