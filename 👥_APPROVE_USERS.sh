#!/bin/bash

# 👥 APPROVE USER SIGN-UP REQUESTS
# Quick script to view and approve pending user requests

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "👥 USER SIGN-UP APPROVAL SYSTEM"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

echo "📋 PENDING REQUESTS:"
echo ""

# Get pending requests
REQUESTS=$(curl -s http://localhost:8001/api/auth/pending-requests)

# Display them nicely
echo "$REQUESTS" | python3 -c "
import sys, json
data = json.load(sys.stdin)
requests = data.get('requests', [])

if not requests:
    print('   ✅ No pending requests!')
    sys.exit(0)

for req in requests:
    print(f\"   ID: {req['id']}\")
    print(f\"   Phone: {req['phone']}\")
    print(f\"   Requested: {req['requested_at']}\")
    print(f\"   ---\")
"

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if there are requests to approve
COUNT=$(echo "$REQUESTS" | python3 -c "import sys, json; print(json.load(sys.stdin).get('count', 0))")

if [ "$COUNT" -eq 0 ]; then
    echo "✅ All caught up! No requests to approve."
    echo ""
    exit 0
fi

echo "🎯 TO APPROVE A USER:"
echo ""
echo "Run this command:"
echo ""
echo "  curl -X POST http://localhost:8001/api/auth/approve-request \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"request_id\": REQUEST_ID_HERE, \"admin_password\": \"rwwc2018\"}'"
echo ""
echo "Example (approve ID 1):"
echo ""
echo "  curl -X POST http://localhost:8001/api/auth/approve-request \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"request_id\": 1, \"admin_password\": \"rwwc2018\"}'"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "💡 TIP: Approve request ID 1 (8472579747) - that's your real user!"
echo ""






