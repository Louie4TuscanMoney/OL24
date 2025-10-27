#!/bin/bash

# Quick restart backend with real-time NBA API fix

echo "🚨 Restarting backend with REAL-TIME NBA API..."
echo ""

# Kill old processes
echo "Stopping old backend..."
pkill -f "trading_dashboard_api" 2>/dev/null
pkill -f "autonomous_trading_daemon" 2>/dev/null
sleep 2

# Restart
echo "Starting NEW backend with real-time fix..."
bash 🚀_START_AUTONOMOUS_SYSTEM.sh

