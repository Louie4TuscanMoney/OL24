#!/bin/bash

# 🚀 ONE-LINE GIT COMMIT & PUSH
# Usage: bash 🚀_GIT_COMMIT_PUSH.sh "your commit message"

# Navigate to dashboard directory
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System/dashboard_pro" || exit

# Ensure we're using HTTPS (not SSH)
git remote set-url origin https://github.com/Louie4TuscanMoney/OL24.git 2>/dev/null

# Check if commit message provided
if [ -z "$1" ]; then
    echo "❌ Error: Provide a commit message!"
    echo "Usage: bash 🚀_GIT_COMMIT_PUSH.sh \"your message\""
    exit 1
fi

COMMIT_MSG="$1"

echo "📦 Staging all changes..."
git add .

echo "📝 Committing: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"

echo "🚀 Pushing to GitHub (OL24)..."
git push origin main

echo ""
echo "✅ COMPLETE!"
echo ""
echo "📊 View on GitHub:"
echo "   https://github.com/Louie4TuscanMoney/OL24"
echo ""
echo "🌐 Deploy to Vercel:"
echo "   1. Go to vercel.com/new"
echo "   2. Import: Louie4TuscanMoney/OL24"
echo "   3. Framework: Vite"
echo "   4. Deploy!"
echo ""

