#!/bin/bash

echo "🚀 DEPLOYING ONTOLOGIC XYZ DASHBOARD TO VERCEL"
echo "=============================================="
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "📦 Installing Vercel CLI..."
    npm install -g vercel
fi

echo "🔐 Logging in to Vercel..."
vercel login

echo ""
echo "🚀 Deploying to production..."
echo ""
echo "Backend API: https://ol24-production.up.railway.app"
echo ""

# Deploy to production
vercel --prod

echo ""
echo "=============================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo ""
echo "🎯 Next steps:"
echo "1. Copy the Vercel URL from above"
echo "2. Open it in your browser"
echo "3. Share with friends!"
echo ""
echo "🔥 Your backend is already live at:"
echo "   https://ol24-production.up.railway.app"
echo ""
echo "=============================================="

