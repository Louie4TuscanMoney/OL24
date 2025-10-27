#!/bin/bash

echo "🔄 AUTO-CHECKING RAILWAY EVERY 30 SECONDS..."
echo "   Press Ctrl+C to stop"
echo ""

while true; do
    TIMESTAMP=$(date +"%H:%M:%S")
    echo "[$TIMESTAMP] Checking Railway..."
    
    RESPONSE=$(curl -s "https://ol24-production.up.railway.app/api/live-games")
    
    if echo "$RESPONSE" | grep -q "System not initialized"; then
        echo "   ⏳ Still initializing..."
    elif echo "$RESPONSE" | grep -q "error"; then
        echo "   ❌ Error: $RESPONSE"
    else
        echo "   ✅ SYSTEM IS WORKING!"
        echo ""
        echo "$RESPONSE" | python3 -m json.tool
        echo ""
        echo "🎉 SUCCESS! System initialized!"
        break
    fi
    
    sleep 30
done

