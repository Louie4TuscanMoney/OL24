#!/bin/bash

echo "🕷️ COPYING ALL BETONLINE FILES..."
echo ""

# Create subdirectories
mkdir -p betonlineofficial/scrapers
mkdir -p betonlineofficial/specifications
mkdir -p betonlineofficial/documentation
mkdir -p betonlineofficial/utilities
mkdir -p betonlineofficial/examples

echo "📂 Copying scrapers..."
# Copy scraper files
cp -v "5. Live System/betonline_live_lines.py" betonlineofficial/scrapers/
cp -v "5. Live System/crawlee_betonline_scraper.py" betonlineofficial/scrapers/
cp -v "5. Live System/implied_probability_calculator.py" betonlineofficial/utilities/

echo ""
echo "📄 Copying specifications..."
# Copy BetOnline specs
cp -v "BETONLINE/WEB_SCRAPE_BETONLINE_SPECS/BETONLINE_IMPLEMENTATION_SPEC.md" betonlineofficial/specifications/ 2>/dev/null || echo "  Spec not found (will create new)"
cp -v "BETONLINE/WEB_SCRAPE_BETONLINE_SPECS/BETONLINE_SCRAPING_REFLECTION.md" betonlineofficial/specifications/ 2>/dev/null || echo "  Reflection not found (will create new)"
cp -v "BETONLINE/README.md" betonlineofficial/specifications/ 2>/dev/null || echo "  README not found (will create new)"

echo ""
echo "✅ BASE FILES COPIED!"
echo "📝 Creating new documentation..."

