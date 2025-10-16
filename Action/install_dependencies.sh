#!/bin/bash
# Install all dependencies for NBA Betting System
# Run this once before testing

echo "============================================"
echo "Installing NBA Betting System Dependencies"
echo "============================================"
echo ""

# Core ML dependencies
echo "📦 Installing ML dependencies..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy>=1.24.0
pip3 install pandas>=2.0.0
pip3 install scipy>=1.10.0
pip3 install scikit-learn>=1.3.0

# Data handling
echo ""
echo "📦 Installing data handling packages..."
pip3 install pyarrow>=12.0.0
pip3 install fastparquet>=2023.4.0

# NBA API
echo ""
echo "📦 Installing NBA API..."
pip3 install nba-api>=1.4.1
pip3 install orjson>=3.9.0
pip3 install aiohttp>=3.9.0

# Web framework
echo ""
echo "📦 Installing web frameworks..."
pip3 install fastapi>=0.100.0
pip3 install uvicorn>=0.23.0
pip3 install pydantic>=2.0.0
pip3 install websockets>=12.0

# Risk management
echo ""
echo "📦 Installing risk management packages..."
pip3 install cvxpy>=1.4.0

# Web scraping (for BetOnline)
echo ""
echo "📦 Installing web scraping tools..."
pip3 install playwright>=1.40.0

# After installing playwright, need to install browsers
echo ""
echo "🌐 Installing Playwright browsers..."
python3 -m playwright install chromium

echo ""
echo "============================================"
echo "✅ Installation Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Run test again: python3 test_system_NOW.py"
echo "2. Test BetOnline scraper manually"
echo "3. Follow 6-Day Plan"

