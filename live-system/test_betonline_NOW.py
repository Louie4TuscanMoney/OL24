"""
QUICK TEST: BetOnline Real Odds Extraction
Run this to verify BetOnline scraping works
"""

import requests
import json
from datetime import datetime


def test_betonline_apis():
    """Test all BetOnline API endpoints"""
    
    print("\n" + "="*80)
    print("🧪 BETONLINE API TEST - FINDING REAL ODDS")
    print("="*80 + "\n")
    
    # API endpoints to try (from BetOnline HTML)
    endpoints = [
        ("Feed API (Live Basketball)", "https://www.betonline.ag/services/feeds/sportsbookv2/betml/event/live/2"),
        ("Offering API (Sport 2)", "https://api-offering.betonline.ag/api/offerings/sport/2/live"),
        ("Offering API (Basketball)", "https://api-offering.betonline.ag/api/offerings/basketball/live"),
        ("External API (Events)", "https://api-offering-ext.betonline.ag/api/events/basketball/live"),
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.betonline.ag/sportsbook/basketball/nba',
        'Origin': 'https://www.betonline.ag',
    }
    
    for name, url in endpoints:
        print(f"🔍 Testing: {name}")
        print(f"   URL: {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Show structure
                    print(f"   Response type: {type(data)}")
                    if isinstance(data, dict):
                        print(f"   Keys: {list(data.keys())[:10]}")
                        
                        # Look for events/games
                        event_keys = ['events', 'games', 'fixtures', 'items', 'data']
                        for key in event_keys:
                            if key in data:
                                events = data[key]
                                if isinstance(events, list) and len(events) > 0:
                                    print(f"   ✅ Found {len(events)} items in '{key}'")
                                    print(f"   First item keys: {list(events[0].keys())[:10] if isinstance(events[0], dict) else 'N/A'}")
                                    
                                    # Save to file for inspection
                                    with open(f'/tmp/betonline_{name.replace(" ", "_")}.json', 'w') as f:
                                        json.dump(data, f, indent=2)
                                    print(f"   📄 Saved to /tmp/betonline_{name.replace(' ', '_')}.json")
                                    break
                    elif isinstance(data, list) and len(data) > 0:
                        print(f"   ✅ Found {len(data)} items in list")
                        if isinstance(data[0], dict):
                            print(f"   First item keys: {list(data[0].keys())[:10]}")
                        
                        # Save to file
                        with open(f'/tmp/betonline_{name.replace(" ", "_")}.json', 'w') as f:
                            json.dump(data, f, indent=2)
                        print(f"   📄 Saved to /tmp/betonline_{name.replace(' ', '_')}.json")
                    
                    print()
                except json.JSONDecodeError:
                    print(f"   ⚠️ Response is not JSON (length: {len(response.text)} chars)")
                    print(f"   First 200 chars: {response.text[:200]}")
                    print()
            else:
                print(f"   ❌ HTTP {response.status_code}")
                print()
                
        except requests.Timeout:
            print(f"   ❌ Timeout after 15s")
            print()
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}")
            print()
    
    print("="*80)
    print("✅ Test complete - check /tmp/ for saved JSON files")
    print("="*80 + "\n")


def test_web_scrape():
    """Test web scraping approach"""
    
    print("\n" + "="*80)
    print("🕸️ BETONLINE WEB SCRAPE TEST")
    print("="*80 + "\n")
    
    url = "https://www.betonline.ag/sportsbook/basketball/nba"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    try:
        print(f"🔍 Fetching: {url}")
        response = requests.get(url, headers=headers, timeout=20)
        
        print(f"Status: {response.status_code}")
        print(f"Content-Length: {len(response.text)} chars")
        
        if response.status_code == 200:
            html = response.text
            
            # Save HTML
            with open('/tmp/betonline_nba_page.html', 'w') as f:
                f.write(html)
            print(f"📄 Saved HTML to /tmp/betonline_nba_page.html")
            
            # Look for data patterns
            patterns = [
                'window.__PRELOADED_STATE__',
                'window.INITIAL_DATA',
                'window.WEBAPP_CONFIG',
                'window.SAS_DATA',
                '"events":',
                '"offerings":',
            ]
            
            print("\n🔍 Searching for data patterns:")
            for pattern in patterns:
                if pattern in html:
                    print(f"   ✅ Found: {pattern}")
                else:
                    print(f"   ❌ Not found: {pattern}")
            
            print("\n✅ HTML saved for manual inspection")
        else:
            print(f"❌ Failed with HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run both tests
    test_betonline_apis()
    test_web_scrape()
    
    print("\n" + "="*80)
    print("📋 NEXT STEPS")
    print("="*80)
    print("1. Check /tmp/betonline_*.json files for data structure")
    print("2. Check /tmp/betonline_nba_page.html for embedded data")
    print("3. Update parser based on actual API response format")
    print("4. Integrate into trading_dashboard_api.py")
    print("="*80 + "\n")

