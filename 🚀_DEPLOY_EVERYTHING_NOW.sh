#!/bin/bash

# ONTOLOGIC XYZ - DEPLOY EVERYTHING IN ONE COMMAND
# This does ALL 3 steps: Start system, Deploy to website, Deploy to Vercel

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🔥 ONTOLOGIC XYZ - DEPLOY EVERYTHING NOW"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "This will:"
echo "  1. Start autonomous system (runs 24/7 in background)"
echo "  2. Deploy dashboard to website repo"
echo "  3. Deploy to Vercel (live on ontologicxyz.com)"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# STEP 1: Start Autonomous System
echo "🚀 STEP 1/3: Starting Autonomous System..."
echo ""
bash 🚀_START_AUTONOMOUS_SYSTEM.sh
echo ""
echo "✅ Autonomous system running in background"
echo ""

# Wait a moment
sleep 3

# STEP 2: Deploy to Website Repo
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📦 STEP 2/3: Deploying to Website Repo..."
echo ""
bash 📦_DEPLOY_TO_WEBSITE.sh
echo ""
echo "✅ Dashboard copied to website repo"
echo ""

# STEP 3: Deploy to Vercel
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🌐 STEP 3/3: Deploying to Vercel..."
echo ""

cd "/Users/test/Desktop/Tuscan Money/Websites/OntologicXYZ.com/NBADashboard" || exit 1

echo "📦 Installing dependencies..."
npm install

echo ""
echo "🚀 Deploying to Vercel..."
echo ""
echo "⚠️ You may need to login to Vercel (vercel login) if not already logged in"
echo ""

vercel deploy --prod

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🎊 COMPLETE DEPLOYMENT FINISHED!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ AUTONOMOUS SYSTEM: Running on your computer (24/7)"
echo "✅ DASHBOARD FILES: Copied to website repo"
echo "✅ VERCEL DEPLOY: Live at ontologicxyz.com/NBADashboard"
echo ""
echo "🎯 System is now:"
echo "   • Monitoring NBA games autonomously"
echo "   • Fetching BetOnline lines automatically"
echo "   • Making ML predictions"
echo "   • Identifying opportunities"
echo "   • Serving data to dashboard"
echo "   • Live on your website"
echo ""
echo "🏀 Access your dashboard at:"
echo "   https://ontologicxyz.com/NBADashboard"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "ALL DONE! 🔥🚀"
echo ""

