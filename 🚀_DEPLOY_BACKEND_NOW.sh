#!/bin/bash

echo "========================================================================"
echo "🚀 DEPLOYING BACKEND TO RAILWAY"
echo "========================================================================"
echo ""

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/live-system"

echo "📍 Current directory: $(pwd)"
echo ""

# Check git status
echo "📊 Checking git status..."
git status
echo ""

# Add all changes
echo "➕ Adding changes..."
git add trading_dashboard_api.py
echo ""

# Commit
echo "💾 Committing..."
git commit -m "Complete backend: player stats, depth charts, conference/division filters"
echo ""

# Push
echo "🚀 Pushing to Railway..."
git push origin main
echo ""

echo "========================================================================"
echo "✅ DEPLOYMENT INITIATED!"
echo "========================================================================"
echo ""
echo "⏳ Railway will deploy in 1-2 minutes..."
echo ""
echo "📝 After deployment, verify with:"
echo ""
echo "   # Should return 15 (only East teams)"
echo "   curl \"https://ol24-production.up.railway.app/api/stats/teams?conference=East\" | jq '.count'"
echo ""
echo "   # Should show conference field"
echo "   curl \"https://ol24-production.up.railway.app/api/stats/teams\" | jq '.teams[0].conference'"
echo ""
echo "========================================================================"

