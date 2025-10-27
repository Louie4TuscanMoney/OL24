#!/bin/bash

# Live System Test Script
# Tests Railway backend + Vercel frontend before live game

API_URL="https://ol24-production.up.railway.app"
FRONTEND_URL="https://ontologicxyz.com"

echo "=================================="
echo "🎯 LIVE SYSTEM TEST"
echo "=================================="
echo ""

echo "1️⃣ Testing Railway Backend Health..."
HEALTH=$(curl -s "$API_URL/")
echo "$HEALTH" | python3 -m json.tool
echo ""

if echo "$HEALTH" | grep -q "online"; then
    echo "✅ Backend is online!"
else
    echo "❌ Backend is not responding!"
    exit 1
fi

echo ""
echo "2️⃣ Testing Live Games Endpoint..."
GAMES=$(curl -s "$API_URL/api/live-games")
echo "$GAMES" | python3 -m json.tool | head -20
echo ""

if echo "$GAMES" | grep -q "error"; then
    echo "⚠️  Backend returned error (may be initializing...)"
    echo "   Check Railway logs: https://railway.app"
else
    echo "✅ Live games endpoint working!"
fi

echo ""
echo "3️⃣ Testing WebSocket Endpoint..."
echo "   WebSocket URL: wss://ol24-production.up.railway.app/ws"
echo "   (Test from browser: Open $FRONTEND_URL and check console)"
echo ""

echo "4️⃣ Testing Opportunities Endpoint..."
OPPS=$(curl -s "$API_URL/api/opportunities")
echo "$OPPS" | python3 -m json.tool | head -20
echo ""

echo "=================================="
echo "✅ BACKEND TEST COMPLETE"
echo "=================================="
echo ""
echo "Next Steps:"
echo "1. Wait for Railway rebuild (~2 minutes)"
echo "2. Check Railway logs for:"
echo "   - '✅ Model downloaded successfully!'"
echo "   - '✅ SYSTEM FULLY INITIALIZED'"
echo "3. Open $FRONTEND_URL"
echo "4. Login and check WebSocket connection"
echo "5. Wait for live game at Q2 6:00"
echo ""
echo "🎯 Railway Logs: https://railway.app/project/YOUR_PROJECT/deployments"
echo ""

