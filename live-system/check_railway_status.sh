#!/bin/bash

echo "🔍 Checking Railway Deployment Status..."
echo ""

# Check backend health
echo "1. Backend Health:"
curl -s "https://ol24-production.up.railway.app/" | python3 -m json.tool
echo ""

# Check if system is initialized
echo ""
echo "2. System Initialization:"
RESPONSE=$(curl -s "https://ol24-production.up.railway.app/api/live-games")
echo "$RESPONSE" | python3 -m json.tool
echo ""

if echo "$RESPONSE" | grep -q "System not initialized"; then
    echo "❌ System still not initialized"
    echo ""
    echo "💡 This means either:"
    echo "   a) Railway is still deploying the fix"
    echo "   b) A new error occurred"
    echo ""
    echo "🔧 Next Steps:"
    echo "   1. Wait 1-2 more minutes"
    echo "   2. Check Railway logs for latest deployment"
    echo "   3. Look for timestamp AFTER 22:40"
    echo ""
elif echo "$RESPONSE" | grep -q "error"; then
    echo "⚠️ Different error occurred"
    echo "$RESPONSE"
else
    echo "✅ SYSTEM IS WORKING!"
    echo ""
    echo "Live games found or system ready for predictions!"
fi

echo ""
echo "📊 Railway Logs: https://railway.app/project/YOUR_PROJECT/deployments"
echo "   Look for deployment with timestamp > 22:40"

