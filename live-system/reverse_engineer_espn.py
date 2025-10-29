#!/usr/bin/env python3
"""
ESPN REVERSE ENGINEERING TOOL
Automatically captures all API calls and WebSocket connections from ESPN.com

This will show you ESPN's REAL-TIME endpoints!
"""

import asyncio
import json
from datetime import datetime

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("❌ Playwright not installed")
    print("   Install with: pip install playwright")
    print("   Then run: playwright install chromium")
    exit(1)


class ESPNReverseEngineer:
    def __init__(self):
        self.websockets = []
        self.api_calls = []
        self.score_updates = []
    
    async def capture_network_traffic(self, duration_seconds=30):
        """
        Opens ESPN scoreboard and captures all network traffic
        
        Args:
            duration_seconds: How long to monitor (default 30s)
        """
        print("="*80)
        print("🕵️  ESPN REVERSE ENGINEERING TOOL")
        print("="*80)
        print(f"Monitoring ESPN.com for {duration_seconds} seconds...")
        print()
        
        async with async_playwright() as p:
            # Launch browser (headless=False to see what's happening)
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            )
            page = await context.new_page()
            
            # CAPTURE WEBSOCKETS
            page.on("websocket", self._on_websocket)
            
            # CAPTURE API CALLS
            page.on("request", self._on_request)
            page.on("response", self._on_response)
            
            print("📡 Opening ESPN NBA Scoreboard...")
            await page.goto('https://www.espn.com/nba/scoreboard', wait_until='networkidle')
            
            print(f"✅ Page loaded! Monitoring for {duration_seconds} seconds...")
            print()
            print("=" * 80)
            print("LIVE NETWORK CAPTURE:")
            print("=" * 80)
            
            # Monitor for specified duration
            await asyncio.sleep(duration_seconds)
            
            await browser.close()
            
            # Print summary
            self._print_summary()
    
    def _on_websocket(self, ws):
        """Called when WebSocket connection is opened"""
        url = ws.url
        print(f"\n🔌 WEBSOCKET DETECTED:")
        print(f"   URL: {url}")
        self.websockets.append({
            'url': url,
            'timestamp': datetime.now().isoformat()
        })
        
        # Listen to WebSocket messages
        ws.on("framereceived", lambda payload: self._on_ws_message(url, payload))
        ws.on("framesent", lambda payload: self._on_ws_sent(url, payload))
    
    def _on_ws_message(self, url, payload):
        """Called when WebSocket receives a message"""
        try:
            text = payload.get('text', '')
            if text:
                print(f"\n📥 WS MESSAGE from {url[:50]}...")
                
                # Try to parse as JSON
                try:
                    data = json.loads(text)
                    print(f"   {json.dumps(data, indent=2)[:200]}...")
                    
                    # Look for score updates
                    if 'score' in str(data).lower() or 'game' in str(data).lower():
                        self.score_updates.append({
                            'url': url,
                            'data': data,
                            'timestamp': datetime.now().isoformat()
                        })
                except:
                    print(f"   Raw: {text[:200]}...")
        except Exception as e:
            print(f"   Error parsing WS message: {e}")
    
    def _on_ws_sent(self, url, payload):
        """Called when WebSocket sends a message"""
        try:
            text = payload.get('text', '')
            if text:
                print(f"\n📤 WS SENT to {url[:50]}...")
                print(f"   {text[:200]}...")
        except:
            pass
    
    def _on_request(self, request):
        """Called when page makes a request"""
        url = request.url
        
        # Filter for relevant APIs only
        keywords = ['score', 'game', 'live', 'realtime', 'stream', 'nba', 'event']
        if any(k in url.lower() for k in keywords):
            if 'espn' in url and not url.endswith(('.png', '.jpg', '.svg', '.css', '.woff')):
                print(f"\n🔍 API REQUEST:")
                print(f"   {request.method} {url}")
                
                self.api_calls.append({
                    'method': request.method,
                    'url': url,
                    'timestamp': datetime.now().isoformat()
                })
    
    def _on_response(self, response):
        """Called when request completes"""
        url = response.url
        
        # Filter for score/game APIs
        keywords = ['score', 'game', 'live', 'realtime', 'stream', 'nba']
        if any(k in url.lower() for k in keywords):
            if 'espn' in url and response.status == 200:
                print(f"✅ API RESPONSE: {response.status} - {url[:80]}...")
    
    def _print_summary(self):
        """Print summary of findings"""
        print("\n\n")
        print("=" * 80)
        print("📊 REVERSE ENGINEERING SUMMARY")
        print("=" * 80)
        
        print(f"\n🔌 WebSockets Found: {len(self.websockets)}")
        for ws in self.websockets:
            print(f"   - {ws['url']}")
        
        print(f"\n📡 API Calls Found: {len(self.api_calls)}")
        unique_urls = list(set(call['url'] for call in self.api_calls))
        for url in unique_urls[:10]:  # Show first 10
            print(f"   - {url}")
        
        print(f"\n⚡ Score Updates Captured: {len(self.score_updates)}")
        if self.score_updates:
            print("   First update:")
            print(f"   {json.dumps(self.score_updates[0], indent=2)[:300]}...")
        
        print("\n" + "=" * 80)
        print("💾 SAVING RESULTS TO FILE...")
        print("=" * 80)
        
        # Save to file
        with open('espn_reverse_engineering_results.json', 'w') as f:
            json.dump({
                'websockets': self.websockets,
                'api_calls': self.api_calls,
                'score_updates': self.score_updates,
                'captured_at': datetime.now().isoformat()
            }, f, indent=2)
        
        print("✅ Results saved to: espn_reverse_engineering_results.json")
        print()
        print("🎯 NEXT STEPS:")
        print("   1. Check the JSON file for WebSocket URLs")
        print("   2. Test connecting to the WebSocket")
        print("   3. Implement in our system!")


async def main():
    """Main entry point"""
    engineer = ESPNReverseEngineer()
    
    # Monitor for 30 seconds (adjust as needed)
    await engineer.capture_network_traffic(duration_seconds=30)
    
    print("\n✅ Capture complete!")


if __name__ == "__main__":
    if not PLAYWRIGHT_AVAILABLE:
        print("Install Playwright first:")
        print("  pip install playwright")
        print("  playwright install chromium")
    else:
        print("\n🚀 Starting ESPN reverse engineering...")
        print("   This will open a browser window - DON'T CLOSE IT!")
        print("   Let it run for 30 seconds to capture network traffic")
        print()
        
        asyncio.run(main())

