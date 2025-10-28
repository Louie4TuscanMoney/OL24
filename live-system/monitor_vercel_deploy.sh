#!/bin/bash

echo "🔄 MONITORING VERCEL DEPLOYMENT..."
echo "Press Ctrl+C to stop"
echo ""

COUNTER=0
while [ $COUNTER -lt 20 ]; do
    TIMESTAMP=$(date +"[%H:%M:%S]")
    echo "$TIMESTAMP Checking Vercel..."
    
    # Check if Vercel is responding
    RESPONSE=$(curl -s -I "https://ontologicxyz.com" 2>&1 | head -1)
    
    if echo "$RESPONSE" | grep -q "200"; then
        echo "   ✅ Vercel is live and responding"
        
        # Try to check if it's the new deployment (this is approximate)
        echo "   → Visit https://ontologicxyz.com to verify WebSocket connection"
        echo "   → Check browser console for: '✅ WebSocket connected!'"
        
    else
        echo "   ⏳ Vercel deploying or unreachable..."
    fi
    
    echo ""
    sleep 15
    COUNTER=$((COUNTER + 1))
done

echo "✅ Monitoring complete. Check https://ontologicxyz.com manually."
