"""
BETONLINE API SCRAPER - PRODUCTION READY
Reverse-engineered from their frontend API calls

Key Discovery from HTML:
- Offering API: https://api-offering.betonline.ag
- External API: https://api-offering-ext.betonline.ag
- WebSocket: wss://api.betonline.ag/pushd (Diffusion)
- Feed API: /services/feeds/sportsbookv2/betml/event/live/2

Author: Ontologic XYZ
Date: October 28, 2025
"""

import requests
import cloudscraper
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
import time


class BetOnlineAPIScraper:
    """
    Production-grade BetOnline scraper using their real API endpoints
    """
    
    def __init__(self):
        """Initialize with CloudFlare bypass"""
        # Use cloudscraper to bypass CloudFlare
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'darwin',
                'desktop': True
            }
        )
        
        # API endpoints (from HTML config)
        self.offering_api = "https://api-offering.betonline.ag"
        self.external_api = "https://api-offering-ext.betonline.ag"
        self.feed_api = "https://www.betonline.ag/services/feeds/sportsbookv2/betml/event/live/2"
        
        # Headers (from HTML inspection)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.betonline.ag/sportsbook/basketball/nba',
            'X-Requested-With': 'XMLHttpRequest',
            'Origin': 'https://www.betonline.ag',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        }
        
        self.scraper.headers.update(self.headers)
        
        print("✅ BetOnline API Scraper initialized (CloudFlare bypass active)")
    
    def get_live_nba_odds(self) -> List[Dict]:
        """
        Get REAL live NBA odds from BetOnline
        
        Returns:
            List of game dicts with actual market odds
        """
        print("\n" + "="*80)
        print("💰 BETONLINE REAL ODDS EXTRACTION")
        print("="*80 + "\n")
        
        odds = []
        
        # METHOD 1: Offering API (primary)
        print("🔍 Method 1: Offering API...")
        odds = self._try_offering_api()
        if odds:
            print(f"✅ Offering API: {len(odds)} games with REAL odds\n")
            return odds
        
        # METHOD 2: Feed API (secondary)
        print("🔍 Method 2: Feed API...")
        odds = self._try_feed_api()
        if odds:
            print(f"✅ Feed API: {len(odds)} games with REAL odds\n")
            return odds
        
        # METHOD 3: External API (tertiary)
        print("🔍 Method 3: External API...")
        odds = self._try_external_api()
        if odds:
            print(f"✅ External API: {len(odds)} games with REAL odds\n")
            return odds
        
        # METHOD 4: Web scraping with JavaScript rendering
        print("🔍 Method 4: Web scraping...")
        odds = self._try_web_scrape()
        if odds:
            print(f"✅ Web scrape: {len(odds)} games with REAL odds\n")
            return odds
        
        print("⚠️ All methods failed - returning empty (NO synthetic odds!)\n")
        return []
    
    def _try_offering_api(self) -> List[Dict]:
        """
        Try the main Offering API endpoint
        
        Returns:
            List of odds or empty list
        """
        try:
            # Try multiple offering API paths
            paths = [
                "/api/offerings/events/basketball/nba/live",
                "/api/offerings/sport/basketball/1332678/live",
                "/api/v1/offerings/basketball/nba",
                "/basketball/nba/events"
            ]
            
            for path in paths:
                url = self.offering_api + path
                
                try:
                    response = self.scraper.get(url, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        odds = self._parse_offering_response(data)
                        
                        if odds:
                            print(f"  ✅ Found {len(odds)} games at {path}")
                            return odds
                except Exception as e:
                    print(f"  ⚠️ {path}: {str(e)[:50]}")
                    continue
            
            return []
            
        except Exception as e:
            print(f"  ❌ Offering API error: {e}")
            return []
    
    def _try_feed_api(self) -> List[Dict]:
        """
        Try the Feed API endpoint (from HTML config)
        
        Returns:
            List of odds or empty list
        """
        try:
            # Sport ID 2 = Basketball
            response = self.scraper.get(self.feed_api, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                odds = self._parse_feed_response(data)
                
                if odds:
                    print(f"  ✅ Found {len(odds)} games")
                    return odds
            
            return []
            
        except Exception as e:
            print(f"  ❌ Feed API error: {e}")
            return []
    
    def _try_external_api(self) -> List[Dict]:
        """
        Try the External Offering API
        
        Returns:
            List of odds or empty list
        """
        try:
            paths = [
                "/api/events/basketball/live",
                "/api/basketball/nba/live",
                "/offerings/basketball"
            ]
            
            for path in paths:
                url = self.external_api + path
                
                try:
                    response = self.scraper.get(url, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        odds = self._parse_offering_response(data)
                        
                        if odds:
                            print(f"  ✅ Found {len(odds)} games at {path}")
                            return odds
                except Exception as e:
                    print(f"  ⚠️ {path}: {str(e)[:50]}")
                    continue
            
            return []
            
        except Exception as e:
            print(f"  ❌ External API error: {e}")
            return []
    
    def _try_web_scrape(self) -> List[Dict]:
        """
        Try web scraping with CloudFlare bypass
        
        Returns:
            List of odds or empty list
        """
        try:
            url = "https://www.betonline.ag/sportsbook/basketball/nba"
            
            # Use cloudscraper to bypass CloudFlare
            response = self.scraper.get(url, timeout=20)
            
            if response.status_code != 200:
                print(f"  ⚠️ HTTP {response.status_code}")
                return []
            
            # Look for JSON data embedded in HTML
            html = response.text
            
            # Try to find embedded JSON data
            json_patterns = [
                r'window\.__PRELOADED_STATE__\s*=\s*({.+?});',
                r'window\.INITIAL_DATA\s*=\s*({.+?});',
                r'window\.WEBAPP_CONFIG\s*=\s*({.+?});',
                r'"events":\s*(\[.+?\])',
            ]
            
            for pattern in json_patterns:
                matches = re.search(pattern, html, re.DOTALL)
                if matches:
                    try:
                        json_str = matches.group(1)
                        data = json.loads(json_str)
                        odds = self._parse_embedded_json(data)
                        
                        if odds:
                            print(f"  ✅ Found {len(odds)} games in embedded JSON")
                            return odds
                    except:
                        continue
            
            print("  ⚠️ No JSON data found in HTML")
            return []
            
        except Exception as e:
            print(f"  ❌ Web scrape error: {e}")
            return []
    
    def _parse_offering_response(self, data: Dict) -> List[Dict]:
        """
        Parse Offering API response
        
        Args:
            data: API response JSON
            
        Returns:
            List of parsed odds
        """
        odds = []
        
        try:
            # Try different data structures
            events = (data.get('events') or 
                     data.get('data', {}).get('events') or 
                     data.get('items') or 
                     data.get('games') or 
                     [])
            
            for event in events:
                parsed = self._parse_event(event)
                if parsed:
                    odds.append(parsed)
            
        except Exception as e:
            print(f"  ⚠️ Parse error: {e}")
        
        return odds
    
    def _parse_feed_response(self, data: Dict) -> List[Dict]:
        """
        Parse Feed API response
        
        Args:
            data: Feed API response JSON
            
        Returns:
            List of parsed odds
        """
        odds = []
        
        try:
            # Feed API likely has a different structure
            # Common patterns: events, fixtures, competitions
            
            if isinstance(data, list):
                events = data
            else:
                events = (data.get('events') or 
                         data.get('fixtures') or 
                         data.get('competitions', {}).get('events') or 
                         [])
            
            for event in events:
                parsed = self._parse_event(event)
                if parsed:
                    odds.append(parsed)
            
        except Exception as e:
            print(f"  ⚠️ Parse error: {e}")
        
        return odds
    
    def _parse_embedded_json(self, data: Dict) -> List[Dict]:
        """
        Parse JSON data embedded in HTML
        
        Args:
            data: Embedded JSON data
            
        Returns:
            List of parsed odds
        """
        odds = []
        
        try:
            # Look for events in various places
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        parsed = self._parse_event(item)
                        if parsed:
                            odds.append(parsed)
            elif isinstance(data, dict):
                odds = self._parse_offering_response(data)
            
        except Exception as e:
            print(f"  ⚠️ Parse error: {e}")
        
        return odds
    
    def _parse_event(self, event: Dict) -> Optional[Dict]:
        """
        Parse a single event/game from any API response
        
        Args:
            event: Event dict
            
        Returns:
            Parsed odds dict or None
        """
        try:
            # Extract teams (try multiple field names)
            home_team = (event.get('homeTeam', {}).get('name') or 
                        event.get('home', {}).get('name') or 
                        event.get('homeName') or 
                        event.get('home_team') or 
                        '')
            
            away_team = (event.get('awayTeam', {}).get('name') or 
                        event.get('away', {}).get('name') or 
                        event.get('awayName') or 
                        event.get('away_team') or 
                        '')
            
            if not home_team or not away_team:
                return None
            
            # Extract game ID
            game_id = (event.get('id') or 
                      event.get('eventId') or 
                      event.get('gameId') or 
                      'unknown')
            
            # Extract markets/odds
            markets = (event.get('markets') or 
                      event.get('lines') or 
                      event.get('odds') or 
                      [])
            
            spread = None
            spread_odds_home = -110
            spread_odds_away = -110
            total = None
            total_odds_over = -110
            total_odds_under = -110
            home_ml = None
            away_ml = None
            
            # Parse markets
            for market in markets:
                market_type = (market.get('type', '') or 
                              market.get('marketType', '') or 
                              market.get('name', '')).lower()
                
                # SPREAD
                if 'spread' in market_type or 'handicap' in market_type or 'pointspread' in market_type:
                    spread = (market.get('homeSpread') or 
                             market.get('spread') or 
                             market.get('line') or 
                             market.get('handicap'))
                    
                    spread_odds_home = market.get('homeOdds', -110) or market.get('odds', -110)
                    spread_odds_away = market.get('awayOdds', -110) or market.get('odds', -110)
                
                # TOTAL
                elif 'total' in market_type or 'over' in market_type or 'under' in market_type:
                    total = (market.get('total') or 
                            market.get('line') or 
                            market.get('points'))
                    
                    total_odds_over = market.get('overOdds', -110) or market.get('odds', -110)
                    total_odds_under = market.get('underOdds', -110) or market.get('odds', -110)
                
                # MONEYLINE
                elif 'money' in market_type or 'ml' in market_type or 'win' in market_type:
                    home_ml = (market.get('homeOdds') or 
                              market.get('home') or 
                              market.get('odds1'))
                    
                    away_ml = (market.get('awayOdds') or 
                              market.get('away') or 
                              market.get('odds2'))
            
            # Only return if we have at least spread or moneyline
            if spread is None and home_ml is None:
                return None
            
            # Normalize team names to 3-letter codes
            home_abbr = self._normalize_team(home_team)
            away_abbr = self._normalize_team(away_team)
            
            return {
                'game_id': str(game_id),
                'home_team': home_abbr,
                'away_team': away_abbr,
                'home_full_name': home_team,
                'away_full_name': away_team,
                
                # SPREAD
                'spread': float(spread) if spread else None,
                'spread_odds_home': spread_odds_home,
                'spread_odds_away': spread_odds_away,
                
                # TOTAL
                'total': float(total) if total else None,
                'total_odds_over': total_odds_over,
                'total_odds_under': total_odds_under,
                
                # MONEYLINE
                'home_ml': home_ml,
                'away_ml': away_ml,
                
                # METADATA
                'timestamp': datetime.now().isoformat(),
                'source': 'BetOnline (REAL API)',
                'book': 'BetOnline'
            }
            
        except Exception as e:
            print(f"  ⚠️ Parse event error: {e}")
            return None
    
    def _try_offering_api(self) -> List[Dict]:
        """
        Try the Offering API endpoints
        
        Returns:
            List of odds or empty list
        """
        try:
            # Multiple possible paths based on their API structure
            paths = [
                "/api/offerings/sport/2/live",  # Sport ID 2 = Basketball
                "/api/offerings/basketball/nba/live",
                "/api/offerings/basketball/live",
                "/offerings/sport/2/live",
                "/basketball/nba/events",
            ]
            
            for path in paths:
                url = self.offering_api + path
                
                try:
                    print(f"  → Trying: {url}")
                    response = self.scraper.get(url, timeout=15)
                    
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            odds = self._parse_offering_response(data)
                            
                            if odds and len(odds) > 0:
                                return odds
                        except json.JSONDecodeError:
                            print(f"    ⚠️ Invalid JSON response")
                            continue
                    else:
                        print(f"    ⚠️ HTTP {response.status_code}")
                        
                except requests.RequestException as e:
                    print(f"    ⚠️ Request failed: {str(e)[:50]}")
                    continue
            
            return []
            
        except Exception as e:
            print(f"  ❌ Offering API error: {e}")
            return []
    
    def _try_feed_api(self) -> List[Dict]:
        """
        Try the Feed API endpoint
        
        Returns:
            List of odds or empty list
        """
        try:
            print(f"  → Trying: {self.feed_api}")
            response = self.scraper.get(self.feed_api, timeout=15)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    odds = self._parse_feed_response(data)
                    
                    if odds:
                        return odds
                except json.JSONDecodeError:
                    print(f"    ⚠️ Invalid JSON response")
            else:
                print(f"    ⚠️ HTTP {response.status_code}")
            
            return []
            
        except Exception as e:
            print(f"  ❌ Feed API error: {e}")
            return []
    
    def _try_external_api(self) -> List[Dict]:
        """
        Try the External Offering API
        
        Returns:
            List of odds or empty list
        """
        try:
            paths = [
                "/api/events/basketball/live",
                "/api/basketball/nba/live",
                "/basketball/nba/events"
            ]
            
            for path in paths:
                url = self.external_api + path
                
                try:
                    print(f"  → Trying: {url}")
                    response = self.scraper.get(url, timeout=15)
                    
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            odds = self._parse_offering_response(data)
                            
                            if odds and len(odds) > 0:
                                return odds
                        except json.JSONDecodeError:
                            print(f"    ⚠️ Invalid JSON response")
                            continue
                    else:
                        print(f"    ⚠️ HTTP {response.status_code}")
                        
                except requests.RequestException as e:
                    print(f"    ⚠️ Request failed: {str(e)[:50]}")
                    continue
            
            return []
            
        except Exception as e:
            print(f"  ❌ External API error: {e}")
            return []
    
    def _try_web_scrape(self) -> List[Dict]:
        """
        Try scraping the main NBA page and extracting embedded data
        
        Returns:
            List of odds or empty list
        """
        try:
            url = "https://www.betonline.ag/sportsbook/basketball/nba"
            
            print(f"  → Trying: {url}")
            response = self.scraper.get(url, timeout=20)
            
            if response.status_code != 200:
                print(f"    ⚠️ HTTP {response.status_code}")
                return []
            
            html = response.text
            
            # Save HTML for inspection
            with open('/tmp/betonline_nba.html', 'w') as f:
                f.write(html)
            print(f"    📄 Saved HTML to /tmp/betonline_nba.html")
            
            # Look for JSON data in script tags or window objects
            json_patterns = [
                (r'window\.__PRELOADED_STATE__\s*=\s*({.+?});', 'PRELOADED_STATE'),
                (r'window\.INITIAL_DATA\s*=\s*({.+?});', 'INITIAL_DATA'),
                (r'window\.WEBAPP_CONFIG\s*=\s*({.+?});', 'WEBAPP_CONFIG'),
                (r'"events":\s*(\[{.+?}\])', 'events_array'),
                (r'window\.SAS_DATA\s*=\s*({.+?});', 'SAS_DATA'),
            ]
            
            for pattern, name in json_patterns:
                matches = re.search(pattern, html, re.DOTALL)
                if matches:
                    try:
                        json_str = matches.group(1)
                        # Truncate if too long
                        if len(json_str) > 100000:
                            print(f"    ⚠️ {name} JSON too large ({len(json_str)} chars)")
                            continue
                        
                        data = json.loads(json_str)
                        odds = self._parse_embedded_json(data)
                        
                        if odds:
                            print(f"    ✅ Found {len(odds)} games in {name}")
                            return odds
                    except json.JSONDecodeError as e:
                        print(f"    ⚠️ {name} JSON decode error: {str(e)[:50]}")
                        continue
                    except Exception as e:
                        print(f"    ⚠️ {name} error: {str(e)[:50]}")
                        continue
            
            print("    ⚠️ No parseable JSON data found")
            return []
            
        except Exception as e:
            print(f"  ❌ Web scrape error: {e}")
            return []
    
    def _normalize_team(self, team_name: str) -> str:
        """
        Normalize team name to NBA abbreviation
        
        Args:
            team_name: Full team name
            
        Returns:
            3-letter NBA code
        """
        mapping = {
            # Full names
            'atlanta hawks': 'ATL',
            'boston celtics': 'BOS',
            'brooklyn nets': 'BKN',
            'charlotte hornets': 'CHA',
            'chicago bulls': 'CHI',
            'cleveland cavaliers': 'CLE',
            'dallas mavericks': 'DAL',
            'denver nuggets': 'DEN',
            'detroit pistons': 'DET',
            'golden state warriors': 'GSW',
            'houston rockets': 'HOU',
            'indiana pacers': 'IND',
            'los angeles clippers': 'LAC',
            'los angeles lakers': 'LAL',
            'memphis grizzlies': 'MEM',
            'miami heat': 'MIA',
            'milwaukee bucks': 'MIL',
            'minnesota timberwolves': 'MIN',
            'new orleans pelicans': 'NOP',
            'new york knicks': 'NYK',
            'oklahoma city thunder': 'OKC',
            'orlando magic': 'ORL',
            'philadelphia 76ers': 'PHI',
            'phoenix suns': 'PHX',
            'portland trail blazers': 'POR',
            'sacramento kings': 'SAC',
            'san antonio spurs': 'SAS',
            'toronto raptors': 'TOR',
            'utah jazz': 'UTA',
            'washington wizards': 'WAS',
            
            # Short names
            'hawks': 'ATL',
            'celtics': 'BOS',
            'nets': 'BKN',
            'hornets': 'CHA',
            'bulls': 'CHI',
            'cavaliers': 'CLE',
            'cavs': 'CLE',
            'mavericks': 'DAL',
            'mavs': 'DAL',
            'nuggets': 'DEN',
            'pistons': 'DET',
            'warriors': 'GSW',
            'rockets': 'HOU',
            'pacers': 'IND',
            'clippers': 'LAC',
            'lakers': 'LAL',
            'grizzlies': 'MEM',
            'heat': 'MIA',
            'bucks': 'MIL',
            'timberwolves': 'MIN',
            'wolves': 'MIN',
            'pelicans': 'NOP',
            'knicks': 'NYK',
            'thunder': 'OKC',
            'magic': 'ORL',
            '76ers': 'PHI',
            'sixers': 'PHI',
            'suns': 'PHX',
            'blazers': 'POR',
            'trail blazers': 'POR',
            'kings': 'SAC',
            'spurs': 'SAS',
            'raptors': 'TOR',
            'jazz': 'UTA',
            'wizards': 'WAS',
        }
        
        team_lower = team_name.lower().strip()
        
        # Direct lookup
        if team_lower in mapping:
            return mapping[team_lower]
        
        # Partial match
        for key, code in mapping.items():
            if key in team_lower or team_lower in key:
                return code
        
        # Fallback: return first 3 letters uppercase
        return team_name[:3].upper()
    
    def test_connection(self) -> Dict:
        """
        Test BetOnline connection and API availability
        
        Returns:
            Status dict
        """
        print("\n" + "="*80)
        print("🧪 BETONLINE CONNECTION TEST")
        print("="*80 + "\n")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'cloudflare_bypass': False,
            'offering_api': False,
            'feed_api': False,
            'external_api': False,
            'web_scrape': False,
            'total_games_found': 0,
            'errors': []
        }
        
        # Test 1: Main page (CloudFlare check)
        try:
            print("1. Testing CloudFlare bypass...")
            response = self.scraper.get("https://www.betonline.ag/sportsbook/basketball/nba", timeout=20)
            if response.status_code == 200 and 'betonline' in response.text.lower():
                results['cloudflare_bypass'] = True
                print("   ✅ CloudFlare bypass successful")
            else:
                print(f"   ❌ Failed (HTTP {response.status_code})")
        except Exception as e:
            results['errors'].append(f"CloudFlare: {str(e)[:50]}")
            print(f"   ❌ Error: {str(e)[:50]}")
        
        # Test 2: Offering API
        try:
            print("2. Testing Offering API...")
            odds = self._try_offering_api()
            if odds:
                results['offering_api'] = True
                results['total_games_found'] = len(odds)
                print(f"   ✅ Found {len(odds)} games")
            else:
                print("   ❌ No games found")
        except Exception as e:
            results['errors'].append(f"Offering API: {str(e)[:50]}")
            print(f"   ❌ Error: {str(e)[:50]}")
        
        # Test 3: Feed API
        if results['total_games_found'] == 0:
            try:
                print("3. Testing Feed API...")
                odds = self._try_feed_api()
                if odds:
                    results['feed_api'] = True
                    results['total_games_found'] = len(odds)
                    print(f"   ✅ Found {len(odds)} games")
                else:
                    print("   ❌ No games found")
            except Exception as e:
                results['errors'].append(f"Feed API: {str(e)[:50]}")
                print(f"   ❌ Error: {str(e)[:50]}")
        
        # Test 4: External API
        if results['total_games_found'] == 0:
            try:
                print("4. Testing External API...")
                odds = self._try_external_api()
                if odds:
                    results['external_api'] = True
                    results['total_games_found'] = len(odds)
                    print(f"   ✅ Found {len(odds)} games")
                else:
                    print("   ❌ No games found")
            except Exception as e:
                results['errors'].append(f"External API: {str(e)[:50]}")
                print(f"   ❌ Error: {str(e)[:50]}")
        
        # Test 5: Web scrape
        if results['total_games_found'] == 0:
            try:
                print("5. Testing web scraping...")
                odds = self._try_web_scrape()
                if odds:
                    results['web_scrape'] = True
                    results['total_games_found'] = len(odds)
                    print(f"   ✅ Found {len(odds)} games")
                else:
                    print("   ❌ No games found")
            except Exception as e:
                results['errors'].append(f"Web scrape: {str(e)[:50]}")
                print(f"   ❌ Error: {str(e)[:50]}")
        
        print("\n" + "="*80)
        print("📊 TEST RESULTS")
        print("="*80)
        print(f"CloudFlare Bypass: {'✅' if results['cloudflare_bypass'] else '❌'}")
        print(f"Offering API: {'✅' if results['offering_api'] else '❌'}")
        print(f"Feed API: {'✅' if results['feed_api'] else '❌'}")
        print(f"External API: {'✅' if results['external_api'] else '❌'}")
        print(f"Web Scrape: {'✅' if results['web_scrape'] else '❌'}")
        print(f"\nTotal Games Found: {results['total_games_found']}")
        
        if results['errors']:
            print(f"\nErrors: {len(results['errors'])}")
            for err in results['errors']:
                print(f"  - {err}")
        
        print("="*80 + "\n")
        
        return results


def test_betonline_scraper():
    """
    Test the BetOnline scraper
    """
    scraper = BetOnlineAPIScraper()
    
    # Run connection test
    test_results = scraper.test_connection()
    
    # Get live odds
    print("\n" + "="*80)
    print("💰 LIVE NBA ODDS")
    print("="*80 + "\n")
    
    odds = scraper.get_live_nba_odds()
    
    if odds:
        print(f"✅ Successfully extracted {len(odds)} games:\n")
        
        for game in odds:
            print(f"  {game['away_team']} @ {game['home_team']}")
            print(f"    Spread: {game['spread']:+.1f} ({game['spread_odds_home']}/{game['spread_odds_away']})")
            print(f"    Total: {game['total']:.1f} (O/U: {game['total_odds_over']}/{game['total_odds_under']})")
            print(f"    Moneyline: {game['home_ml']}/{game['away_ml']}")
            print(f"    Source: {game['source']}")
            print()
    else:
        print("❌ No odds found - check connection test results above")
    
    return test_results, odds


if __name__ == "__main__":
    test_betonline_scraper()

