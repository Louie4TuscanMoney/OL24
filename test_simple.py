#!/usr/bin/env python3

print("Testing basic functionality...")

try:
    import cloudscraper
    print("✅ cloudscraper imported")
    
    scraper = cloudscraper.create_scraper()
    print("✅ scraper created")
    
    # Test a simple request
    response = scraper.get("https://httpbin.org/get", timeout=10)
    print(f"✅ Test request: {response.status_code}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")
