#!/usr/bin/env python3

import os
import sys
import time
import pandas as pd
import psycopg2
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import cloudscraper
        print("✅ cloudscraper")
    except Exception as e:
        print(f"❌ cloudscraper: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas")
    except Exception as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        import psycopg2
        print("✅ psycopg2")
    except Exception as e:
        print(f"❌ psycopg2: {e}")
        return False
    
    return True

def test_database():
    """Test database connection"""
    print("\nTesting database...")
    
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set")
        return False
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM players")
        player_count = cur.fetchone()[0]
        print(f"✅ Database connected: {player_count} players")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_web_scraping():
    """Test web scraping"""
    print("\nTesting web scraping...")
    
    try:
        import cloudscraper
        from io import StringIO
        
        scraper = cloudscraper.create_scraper()
        
        # Test with a simple site first
        response = scraper.get("https://httpbin.org/get", timeout=10)
        print(f"✅ Test request: {response.status_code}")
        
        # Now try Basketball Reference
        url = "https://www.basketball-reference.com/leagues/NBA_2026_per_poss.html"
        print(f"Fetching: {url}")
        
        response = scraper.get(url, timeout=30)
        print(f"✅ BR request: {response.status_code}, {len(response.content)} bytes")
        
        # Try to parse with pandas
        dfs = pd.read_html(StringIO(response.text))
        print(f"✅ Found {len(dfs)} tables")
        
        if len(dfs) > 0:
            df = dfs[0]
            print(f"✅ Table shape: {df.shape}")
            print(f"✅ Columns: {list(df.columns)[:5]}")
            
            # Show first few players
            for i in range(min(3, len(df))):
                row = df.iloc[i]
                player = str(row.get('Player', row.iloc[1] if len(row) > 1 else '')).strip()
                if player and player != 'Player':
                    print(f"  Player {i+1}: {player}")
        
        return True
        
    except Exception as e:
        print(f"❌ Web scraping error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("🏀 BASKETBALL REFERENCE SCRAPER TEST")
    print("="*60)
    
    if not test_imports():
        print("\n❌ Import test failed")
        return
    
    if not test_database():
        print("\n❌ Database test failed")
        return
    
    if not test_web_scraping():
        print("\n❌ Web scraping test failed")
        return
    
    print("\n✅ All tests passed!")
    print("\nNext: Run the full scraper")

if __name__ == "__main__":
    main()

