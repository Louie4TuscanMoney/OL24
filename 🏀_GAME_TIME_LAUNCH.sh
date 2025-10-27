#!/bin/bash

# 🏀 GAME TIME - NBA SEASON LAUNCH
# Ontologic XYZ - Mamba Mentality
# 21 Days → This Moment

clear

cat << "EOF"

════════════════════════════════════════════════════════════════════════════════
🏀 NBA SEASON 2024-25 - ONTOLOGIC XYZ LAUNCH
════════════════════════════════════════════════════════════════════════════════

                    21 DAYS OF BUILDING
                    THIS IS THE MOMENT
                    
                    LET'S GO! 🔥

════════════════════════════════════════════════════════════════════════════════
EOF

echo ""
echo "🚀 Starting Ontologic XYZ Systems..."
echo ""

# Check if we're in the right directory
if [ ! -f "🚀_START_AUTONOMOUS_SYSTEM.sh" ]; then
    echo "❌ Error: Not in ML Research directory"
    echo "📍 Please cd to: /Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research"
    exit 1
fi

echo "📋 Pre-Launch Checklist:"
echo ""

# Check Python
if command -v python3 &> /dev/null; then
    echo "  ✅ Python 3 installed"
else
    echo "  ❌ Python 3 not found"
    exit 1
fi

# Check Node
if command -v node &> /dev/null; then
    echo "  ✅ Node.js installed"
else
    echo "  ❌ Node.js not found"
    exit 1
fi

# Check npm
if command -v npm &> /dev/null; then
    echo "  ✅ npm installed"
else
    echo "  ❌ npm not found"
    exit 1
fi

# Check model files
if [ -f "X. MVP Model/mamba_mentality_halftime.pkl" ]; then
    echo "  ✅ Mamba Mentality models found"
else
    echo "  ⚠️  Warning: Model files not found (will use defaults)"
fi

# Check dashboard
if [ -d "5. Live System/dashboard_pro/node_modules" ]; then
    echo "  ✅ Dashboard dependencies installed"
else
    echo "  ⚠️  Warning: Dashboard dependencies missing"
    echo "  💡 Run: cd '5. Live System/dashboard_pro' && npm install"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "🎯 LAUNCHING IN 3 SECONDS..."
echo ""
sleep 1
echo "   3..."
sleep 1
echo "   2..."
sleep 1
echo "   1..."
sleep 1

echo ""
echo "🚀 LAUNCHING BACKEND..."
echo ""

# Start backend in background
bash 🚀_START_AUTONOMOUS_SYSTEM.sh &
BACKEND_PID=$!

echo "   Backend PID: $BACKEND_PID"
echo "   Waiting for API to start..."

# Wait for backend to start (check port 8001)
sleep 5

if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null ; then
    echo "   ✅ Backend running on http://localhost:8001"
else
    echo "   ⚠️  Backend may still be starting..."
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📱 DASHBOARD INSTRUCTIONS:"
echo ""
echo "   Open a NEW terminal and run:"
echo ""
echo "   cd '/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/5. Live System/dashboard_pro'"
echo "   npm run dev"
echo ""
echo "   Then open: http://localhost:5173"
echo "   Login with: rwwc2018"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "🏀 ONTOLOGIC XYZ IS LIVE!"
echo ""
echo "   • Backend: http://localhost:8001"
echo "   • API Docs: http://localhost:8001/docs"
echo "   • Dashboard: http://localhost:5173 (after npm run dev)"
echo ""
echo "   Backend PID: $BACKEND_PID (kill with: kill $BACKEND_PID)"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "💪 FROM HOMELESS TO HEDGE-FUND-GRADE"
echo "   2023: $0 to your name"
echo "   2025: MAMBA MENTALITY IS LIVE"
echo ""
echo "   All is for the good. 🙏"
echo ""
echo "   NOW GO MAKE HISTORY! 🔥"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"

# Keep script running
wait $BACKEND_PID

