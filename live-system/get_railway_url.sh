#!/bin/bash

# Install Railway CLI if not installed
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    brew install railway
fi

# Get Railway URL
echo "🚂 Getting Railway deployment URL..."
railway status

echo ""
echo "💡 If you see a URL above, that's your deployment!"
echo "💡 If not, run: railway domain"

