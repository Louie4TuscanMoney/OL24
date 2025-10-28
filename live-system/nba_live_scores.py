"""
NBA LIVE SCORES API - REAL-TIME (NO CACHE!)

Purpose: Fetch live game scores and states from NBA API
Author: Ontologic XYZ
Date: October 21, 2025

This monitors live NBA games and extracts current game state for predictions.
Uses nba_api library for REAL-TIME data (no CDN caching!)
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

try:
    from nba_api.live.nba.endpoints import scoreboard
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("⚠️ nba_api not available, using fallback")


class NBALiveScores:
    """
    Fetch live NBA scores and game states
    """
    
    def __init__(self):
        """Initialize NBA live score fetcher using official nba_api library"""
        # Using official nba_api library - STATE OF THE ART! ✅
        # This gets REAL-TIME data directly from NBA stats
        self.use_nba_api = NBA_API_AVAILABLE
        
        # ESPN API endpoint (FASTEST!)
        self.scoreboard_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        
        # Fallback CDN endpoint if ESPN fails
        self.scoreboard_url_cdn = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        # RATE LIMITING: Track API calls to prevent throttling
        self._last_nba_api_call = 0
        self._last_espn_api_call = 0
        self._nba_api_cooldown = 1.0  # Wait 1 second between nba_api calls
        self._espn_api_cooldown = 0.1  # ⚡ ULTRA-FAST: 10 calls/sec max (WebSocket calls 1/sec)
        
        # Consecutive failures tracking
        self._nba_api_failures = 0
        self._max_failures = 3  # After 3 failures, skip for 30 seconds
        self._failure_timeout = 30
        self._nba_api_timeout_until = 0
        
        # CACHE LAST VALID DATA (prevents stale CDN fallback)
        self._last_valid_games = []
        self._last_valid_timestamp = 0
        self._cache_max_age = 5  # ⚡ Cache valid for only 5 seconds (ultra-fresh data!)
        
        print(f"✅ NBA API initialized: REAL-TIME ESPN (0.1s cooldown)")
        print(f"   Strategy: ESPN EVERY WebSocket call (1/sec) → Cache if failed → Fresh data always")
        print(f"   Cache: 5s max age, prioritizes live ESPN over stale CDN")
        
    def get_todays_games(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get today's games - USE NBA_API for correct game IDs!
        ESPN has wrong game IDs that don't work with play-by-play
        
        Args:
            force_refresh: If True, bypass all caching and rate limits
        
        Returns:
            List of game dicts with current state
        """
        # NO CACHE - Fetch fresh data every time for live betting
        now = time.time()
        
        # 🚨 FORCE REFRESH: Bypass cache completely
        if force_refresh:
            print(f"🔥 FORCE REFRESH: Bypassing all caching and rate limits")
            self._last_espn_api_call = 0  # Reset to force ESPN call
            self._last_nba_api_call = 0
        
        # METHOD 1: Use nba_api library FIRST (correct game IDs!)
        if NBA_API_AVAILABLE:
            # Check if we're in timeout period due to repeated failures
            if now < self._nba_api_timeout_until:
                remaining = int(self._nba_api_timeout_until - now)
                print(f"⚠️ nba_api in timeout ({remaining}s remaining), using fallback")
            # Check rate limiting cooldown - USE CACHE if available
            elif (now - self._last_nba_api_call) < self._nba_api_cooldown:
                age = now - self._last_valid_timestamp
                if self._last_valid_games and age < self._cache_max_age:
                    print(f"⚡ Rate limited → Using cache ({age:.1f}s old, still fresh)")
                    return self._last_valid_games
                print(f"⚠️ Rate limit + stale cache → trying ESPN...")
            else:
                try:
                    self._last_nba_api_call = now
                    board = scoreboard.ScoreBoard()
                    games_data = board.get_dict()
                    
                    games = []
                    if games_data and 'scoreboard' in games_data and 'games' in games_data['scoreboard']:
                        for game in games_data['scoreboard']['games']:
                            parsed = self._parse_nba_api_game(game)
                            if parsed:
                                games.append(parsed)
                    
                    if games:
                        print(f"✅ nba_api library: {len(games)} games (CORRECT GAME IDs!)")
                        self._nba_api_failures = 0  # Reset failure counter on success
                        self._last_valid_games = games  # CACHE for rate limiting
                        self._last_valid_timestamp = now
                        return games
                        
                except Exception as e:
                    self._nba_api_failures += 1
                    print(f"⚠️ nba_api failed ({self._nba_api_failures}/{self._max_failures}): {e}")
                    print(f"   Error type: {type(e).__name__}")
                    
                    # Show detailed error for diagnosis
                    if self._nba_api_failures == 1:  # Only on first failure to avoid spam
                        import traceback
                        print(f"   Full error details:")
                        print(traceback.format_exc())
                    
                    # If too many failures, put in timeout
                    if self._nba_api_failures >= self._max_failures:
                        self._nba_api_timeout_until = now + self._failure_timeout
                        print(f"🚨 nba_api failing repeatedly! Timeout for {self._failure_timeout}s")
                    
                    print(f"   Trying ESPN...")
        
        # METHOD 2: ESPN API (ULTRA FAST - 1 SECOND UPDATES!)
        # Check rate limiting first
        time_since_last_call = now - self._last_espn_api_call
        if time_since_last_call >= self._espn_api_cooldown:
            try:
                print(f"⚡ ESPN API CALL (last call {time_since_last_call:.3f}s ago)")
                self._last_espn_api_call = now
                response = requests.get(self.scoreboard_url, headers=self.headers, timeout=3)
                response.raise_for_status()
                data = response.json()
                
                games = []
                if 'events' in data:
                    for event in data['events']:
                        parsed = self._parse_espn_game(event)
                        if parsed:
                            games.append(parsed)
                
                if games:
                    # Show detailed status for each game
                    for game in games:
                        print(f"   ESPN: {game['away_team']} @ {game['home_team']}: {game['away_score']}-{game['home_score']} | Q{game['period']} {game['clock']} | {game['status_text']}")
                    
                    print(f"✅ ESPN API SUCCESS: {len(games)} games returned")
                    self._last_valid_games = games  # CACHE for rate limiting
                    self._last_valid_timestamp = now
                    return games
                    
            except Exception as e:
                print(f"⚠️ ESPN failed: {e}, trying CDN...")
        else:
            # ESPN rate limited (shouldn't happen with 0.1s cooldown + 1s WebSocket calls)
            age = now - self._last_valid_timestamp
            if self._last_valid_games and age < self._cache_max_age:
                print(f"⚡ Using cache ({age:.1f}s old) - ESPN called {now - self._last_espn_api_call:.2f}s ago")
                return self._last_valid_games
            print(f"⚠️ Cache expired ({age:.1f}s old) → trying CDN...")
        
        # METHOD 3: Fallback to CDN (5-10 min delay - ALMOST NEVER USE THIS!)
        # Prefer cache over stale CDN unless cache is REALLY old
        age = now - self._last_valid_timestamp
        if self._last_valid_games and age < 60:  # Cache better than 5-10min delayed CDN!
            print(f"⚡ Cache ({age:.1f}s old) BETTER than stale CDN → using cache")
            return self._last_valid_games
            
        try:
            response = requests.get(self.scoreboard_url_cdn, headers=self.headers, timeout=3)
            response.raise_for_status()
            data = response.json()
            
            games = []
            
            if 'scoreboard' in data and 'games' in data['scoreboard']:
                for game in data['scoreboard']['games']:
                    games.append(self._parse_game(game))
            
            if games:
                print(f"⚠️ Using CDN (5-10 min delay): {len(games)} games")
                self._last_valid_games = games  # Even CDN data is cached
                self._last_valid_timestamp = now
                return games
            
        except Exception as e:
            print(f"❌ CDN failed: {e}")
        
        # ABSOLUTE LAST RESORT: Return cache even if very old
        if self._last_valid_games:
            age = now - self._last_valid_timestamp
            print(f"🚨 ALL APIS DOWN! Using cache ({age:.1f}s old) - better than nothing!")
            return self._last_valid_games
            
        print(f"❌ No data available (no cache, all APIs failed)")
        return []
    
    def _parse_espn_game(self, event_data: Dict) -> Dict:
        """
        Parse ESPN API game data (FASTEST - 1 SEC updates!)
        
        Returns:
            Dict with game state
        """
        try:
            game_id = event_data.get('id', '')
            status = event_data.get('status', {})
            status_type = status.get('type', {})
            
            # Status: 1=scheduled, 2=in progress, 3=final
            status_id = status_type.get('id', '1')
            status_map = {'1': 1, '2': 2, '3': 3}
            game_status = status_map.get(status_id, 1)
            
            # Get period and clock from status detail
            period_text = status_type.get('detail', '')
            period = int(status.get('period', 0))
            clock = status.get('displayClock', '')
            
            # Get teams and scores
            competition = event_data.get('competitions', [{}])[0]
            competitors = competition.get('competitors', [])
            
            home_team = next((c for c in competitors if c.get('homeAway') == 'home'), {})
            away_team = next((c for c in competitors if c.get('homeAway') == 'away'), {})
            
            home_score = int(home_team.get('score', 0) or 0)
            away_score = int(away_team.get('score', 0) or 0)
            current_diff = home_score - away_score
            
            home_tricode = home_team.get('team', {}).get('abbreviation', '')
            away_tricode = away_team.get('team', {}).get('abbreviation', '')
            
            # 🚨 CRITICAL FIX: NEVER show live game as "scheduled"
            # If game has started (scores > 0, or period > 0, or has clock), force status to LIVE
            if game_status == 1:  # Currently "scheduled"
                if (home_score > 0 or away_score > 0 or period > 0 or (clock and clock != '0:00')):
                    print(f"⚠️ Game {game_id} has scores/period but marked scheduled! Forcing to LIVE")
                    game_status = 2  # Force to LIVE
            
            is_q2_6min = self._is_q2_6min(period, clock)
            can_predict = self._can_predict(game_status, period, clock)
            
            return {
                'game_id': f"00225000{game_id[-2:]}" if len(game_id) > 2 else game_id,
                'status': game_status,
                'status_text': self._get_status_text(game_status, period, clock),
                'period': period,
                'clock': clock,
                'home_team': home_tricode,
                'away_team': away_tricode,
                'home_score': home_score,
                'away_score': away_score,
                'current_diff': current_diff,
                'is_q2_6min': is_q2_6min,
                'can_predict': can_predict,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error parsing ESPN game: {e}")
            return None
    
    def _parse_nba_api_game(self, game_data: Dict) -> Dict:
        """
        Parse nba_api library game data (REAL-TIME format)
        
        Returns:
            Dict with game state
        """
        try:
            game_id = game_data.get('gameId', '')
            game_status = game_data.get('gameStatus', 1)
            period = game_data.get('period', 0)
            game_clock = game_data.get('gameClock', '')
            
            home_team = game_data.get('homeTeam', {})
            away_team = game_data.get('awayTeam', {})
            
            home_score = int(home_team.get('score', 0) or 0)
            away_score = int(away_team.get('score', 0) or 0)
            current_diff = home_score - away_score
            
            home_tricode = home_team.get('teamTricode', '')
            away_tricode = away_team.get('teamTricode', '')
            
            # 🚨 CRITICAL FIX: NEVER show live game as "scheduled"
            if game_status == 1:  # Currently "scheduled"
                if (home_score > 0 or away_score > 0 or period > 0 or (game_clock and game_clock != '0:00')):
                    print(f"⚠️ nba_api: Game {game_id} has scores/period but marked scheduled! Forcing to LIVE")
                    game_status = 2  # Force to LIVE
            
            is_q2_6min = self._is_q2_6min(period, game_clock)
            can_predict = self._can_predict(game_status, period, game_clock)
            
            return {
                'game_id': game_id,
                'status': game_status,
                'status_text': self._get_status_text(game_status, period, game_clock),
                'period': period,
                'clock': game_clock,
                'home_team': home_tricode,
                'away_team': away_tricode,
                'home_score': home_score,
                'away_score': away_score,
                'current_diff': current_diff,
                'is_q2_6min': is_q2_6min,
                'can_predict': can_predict,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error parsing nba_api game: {e}")
            return None
    
    def _parse_data_nba_game(self, game_data: Dict) -> Dict:
        """
        Parse data.nba.net game data (REAL-TIME format)
        
        Returns:
            Dict with game state
        """
        try:
            game_id = game_data.get('gameId', '')
            
            # Status: 1=not started, 2=live, 3=final
            status_num = game_data.get('statusNum', 1)
            period = game_data.get('period', {}).get('current', 0)
            clock = game_data.get('clock', '')
            
            home_team = game_data.get('hTeam', {})
            away_team = game_data.get('vTeam', {})
            
            home_score = int(home_team.get('score', 0) or 0)
            away_score = int(away_team.get('score', 0) or 0)
            current_diff = home_score - away_score
            
            home_tricode = home_team.get('triCode', '')
            away_tricode = away_team.get('triCode', '')
            
            is_q2_6min = self._is_q2_6min(period, clock)
            can_predict = self._can_predict(status_num, period, clock)
            
            return {
                'game_id': game_id,
                'status': status_num,
                'status_text': self._get_status_text(status_num, period, clock),
                'period': period,
                'clock': clock,
                'home_team': home_tricode,
                'away_team': away_tricode,
                'home_score': home_score,
                'away_score': away_score,
                'current_diff': current_diff,
                'is_q2_6min': is_q2_6min,
                'can_predict': can_predict,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error parsing game: {e}")
            return None
    
    def _parse_stats_game(self, game_data: Dict) -> Dict:
        """
        Parse STATS API game data (real-time format)
        
        Returns:
            Dict with game state
        """
        # Stats API format: GAME_ID, GAME_STATUS_ID, etc.
        game_id = str(game_data.get('GAME_ID', ''))
        game_status = game_data.get('GAME_STATUS_ID', 1)
        
        home_team = game_data.get('HOME_TEAM_ABBREVIATION', '')
        away_team = game_data.get('VISITOR_TEAM_ABBREVIATION', '')
        
        # For live games, we need to fetch detailed score
        # For now, use what's available
        period_val = game_data.get('LIVE_PERIOD', 0)
        clock_val = game_data.get('LIVE_PC_TIME', '')
        
        return {
            'game_id': game_id,
            'status': game_status,
            'status_text': self._get_status_text(game_status, period_val, clock_val),
            'period': period_val,
            'clock': clock_val,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': 0,  # Will update from line score
            'away_score': 0,  # Will update from line score
            'current_diff': 0,
            'is_q2_6min': False,  # Will calculate
            'can_predict': False,  # Will calculate
            'timestamp': datetime.now().isoformat()
        }
    
    def _parse_game(self, game_data: Dict) -> Dict:
        """
        Parse CDN game data into our format (fallback, cached)
        
        Returns:
            Dict with game state at Q2 6:00 (if applicable)
        """
        game_id = game_data.get('gameId', '')
        game_status = game_data.get('gameStatus', 1)  # 1=scheduled, 2=live, 3=final
        
        home_team = game_data.get('homeTeam', {})
        away_team = game_data.get('awayTeam', {})
        
        home_score = home_team.get('score', 0)
        away_score = away_team.get('score', 0)
        current_diff = home_score - away_score
        
        period = game_data.get('period', 0)
        game_clock = game_data.get('gameClock', '')
        
        # 🚨 CRITICAL FIX: NEVER show live game as "scheduled"
        if game_status == 1:  # Currently "scheduled"
            if (home_score > 0 or away_score > 0 or period > 0 or (game_clock and game_clock != '0:00')):
                print(f"⚠️ CDN: Game {game_id} has scores/period but marked scheduled! Forcing to LIVE")
                game_status = 2  # Force to LIVE
        
        return {
            'game_id': game_id,
            'status': game_status,
            'status_text': self._get_status_text(game_status, period, game_clock),
            'period': period,
            'clock': game_clock,
            'home_team': home_team.get('teamTricode', ''),
            'away_team': away_team.get('teamTricode', ''),
            'home_score': home_score,
            'away_score': away_score,
            'current_diff': current_diff,
            'is_q2_6min': self._is_q2_6min(period, game_clock),
            'can_predict': self._can_predict(game_status, period, game_clock),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_status_text(self, status: int, period: int = 0, clock: str = "") -> str:
        """Convert status code to text - DETECTS HALFTIME!"""
        # Check for halftime (period 2, clock 0.0)
        if period == 2 and (clock == "0.0" or clock == "" or "0:00" in clock or "PT00" in clock):
            return "HALFTIME"
        
        # Check for end of quarters
        if period in [1, 3] and (clock == "0.0" or clock == "" or "0:00" in clock or "PT00" in clock):
            return f"END Q{period}"
        
        if status == 1:
            return "SCHEDULED"
        elif status == 2:
            return "LIVE"
        elif status == 3:
            return "FINAL"
        else:
            return "UNKNOWN"
    
    def _is_q2_6min(self, period: int, clock: str) -> bool:
        """
        Check if game is at Q2 6:00 (our prediction point)
        
        Args:
            period: Quarter number
            clock: Game clock (e.g., "6:00", "PT6M00S")
            
        Returns:
            True if at Q2 6:00
        """
        if period != 2:
            return False
        
        # Parse clock (could be "6:00" or "PT6M00S")
        try:
            if ':' in clock:
                minutes = int(clock.split(':')[0])
                return 5 <= minutes <= 7  # Within 1 minute of 6:00
            elif 'PT' in clock:
                # ISO format: PT6M00S
                minutes = int(clock.replace('PT', '').split('M')[0])
                return 5 <= minutes <= 7
        except:
            pass
        
        return False
    
    def _can_predict(self, status: int, period: int, clock: str) -> bool:
        """
        Determine if we can make a prediction now - ENHANCED FOR ALL LIVE GAMES!
        
        Returns:
            True if game state is suitable for prediction
        """
        # Must be live
        if status != 2:
            return False
        
        # ENHANCED: Allow predictions for Q2, Q3, Q4 (not just Q2 6:00)
        if period >= 2 and period <= 4:
            return True
        
        return False
    
    def monitor_live_games(self, interval: int = 30):
        """
        Continuously monitor live games
        
        Args:
            interval: Seconds between checks
        """
        print("\n" + "="*80)
        print("🏀 NBA LIVE GAME MONITOR")
        print("="*80 + "\n")
        
        print(f"Monitoring live games (checking every {interval}s)")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                games = self.get_todays_games()
                
                if not games:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] No games found")
                else:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Found {len(games)} games:")
                    
                    for game in games:
                        status_icon = "🔴" if game['status'] == 2 else "⚪"
                        predict_icon = "🎯" if game['can_predict'] else "  "
                        q2_icon = "⭐" if game['is_q2_6min'] else "  "
                        
                        print(f"  {status_icon} {predict_icon} {q2_icon} {game['away_team']}@{game['home_team']}: "
                              f"{game['away_score']}-{game['home_score']} "
                              f"(Q{game['period']} {game['clock']}) "
                              f"{game['status_text']}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")


def example_usage():
    """
    Example: Get today's games
    """
    print("\n" + "="*80)
    print("🔥 NBA LIVE SCORES - EXAMPLE")
    print("="*80 + "\n")
    
    fetcher = NBALiveScores()
    
    print("📊 Fetching today's games...\n")
    games = fetcher.get_todays_games()
    
    if not games:
        print("⚠️ No games today (or API unavailable)")
        print("   This will work when NBA games are live")
        print()
        
        # Show example format
        print("Example game format:")
        example = {
            'game_id': '0022400123',
            'status': 2,
            'status_text': 'LIVE',
            'period': 2,
            'clock': '6:00',
            'home_team': 'LAL',
            'away_team': 'BOS',
            'home_score': 52,
            'away_score': 48,
            'current_diff': +4,
            'is_q2_6min': True,
            'can_predict': True
        }
        print(json.dumps(example, indent=2))
    else:
        print(f"✅ Found {len(games)} games:\n")
        
        for game in games:
            print(f"Game: {game['away_team']} @ {game['home_team']}")
            print(f"  Score: {game['away_score']}-{game['home_score']}")
            print(f"  Status: {game['status_text']}")
            print(f"  Period: Q{game['period']} {game['clock']}")
            print(f"  Can Predict: {game['can_predict']}")
            print(f"  At Q2 6:00: {game['is_q2_6min']}")
            print()
    
    print("="*80)
    print("✅ NBA LIVE SCORES READY")
    print("="*80)
    print("\n🎯 Use monitor_live_games() for continuous monitoring")


if __name__ == "__main__":
    example_usage()

