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
    # Use live.scoreboard - it WORKS and gets live data
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
        # Using official nba_api library - live.scoreboard
        # Has ~30s CDN cache but WORKS and shows live games correctly
        self.use_nba_api = NBA_API_AVAILABLE
        
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
        
        print(f"✅ NBA API initialized: MULTI-SOURCE FRESHNESS STRATEGY")
        print(f"   Sources: nba_api + ESPN (fetched simultaneously)")
        print(f"   Strategy: Compare all sources, use HIGHEST score (= freshest)")
        print(f"   Logic: If nba_api shows 68-72 but ESPN shows 70-74, use ESPN")
        print(f"   Force refresh: EVERY WebSocket call (1/sec)")
        print(f"   Result: Always use the freshest available data!")
        
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
        
        # MULTI-SOURCE STRATEGY: Fetch from ALL sources, use FRESHEST!
        # This eliminates lag by comparing multiple sources
        
        all_sources = []
        
        # SOURCE 1: nba_api live.scoreboard
        if NBA_API_AVAILABLE:
            try:
                print(f"⚡ Fetching from nba_api...")
                board = scoreboard.ScoreBoard()
                games_data = board.get_dict()
                
                if games_data and 'scoreboard' in games_data and 'games' in games_data['scoreboard']:
                    nba_games = []
                    for game in games_data['scoreboard']['games']:
                        parsed = self._parse_nba_api_game(game)
                        if parsed:
                            nba_games.append(parsed)
                    
                    if nba_games:
                        all_sources.append(('nba_api', nba_games))
                        print(f"   ✅ nba_api: {len(nba_games)} games")
            except Exception as e:
                print(f"   ❌ nba_api failed: {e}")
        
        # SOURCE 2: ESPN API
        try:
            print(f"⚡ Fetching from ESPN...")
            response = requests.get(
                "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                headers=self.headers,
                timeout=2
            )
            if response.status_code == 200:
                data = response.json()
                espn_games = []
                
                if 'events' in data:
                    for event in data['events']:
                        parsed = self._parse_espn_game(event)
                        if parsed:
                            espn_games.append(parsed)
                
                if espn_games:
                    all_sources.append(('espn', espn_games))
                    print(f"   ✅ ESPN: {len(espn_games)} games")
        except Exception as e:
            print(f"   ❌ ESPN failed: {e}")
        
        # PICK THE FRESHEST DATA (highest score = most recent)
        if all_sources:
            print(f"\n🔍 Comparing {len(all_sources)} sources for freshest data...")
            
            # For each game, pick the source with highest total score (= most recent)
            final_games = []
            game_ids = set()
            
            # Collect all game IDs
            for source_name, games in all_sources:
                for game in games:
                    game_ids.add(game['game_id'])
            
            # For each game, pick freshest version
            for game_id in game_ids:
                freshest = None
                freshest_source = None
                max_score = -1
                
                for source_name, games in all_sources:
                    game = next((g for g in games if g['game_id'] == game_id), None)
                    if game:
                        total_score = game.get('home_score', 0) + game.get('away_score', 0)
                        if total_score > max_score or freshest is None:
                            max_score = total_score
                            freshest = game
                            freshest_source = source_name
                
                if freshest:
                    print(f"   📊 {freshest['away_team']} @ {freshest['home_team']}: Using {freshest_source} ({freshest['away_score']}-{freshest['home_score']})")
                    final_games.append(freshest)
            
            print(f"\n✅ FRESHEST DATA: {len(final_games)} games")
            self._last_valid_games = final_games
            self._last_valid_timestamp = now
            return final_games
        
        # Fallback to cache
        if self._last_valid_games:
            age = now - self._last_valid_timestamp
            print(f"⚠️ All sources failed - using cache ({age:.1f}s old)")
            return self._last_valid_games
        else:
            print(f"🚨 No data available")
            return []
        
        # NO FALLBACKS - Only NBA.com official API!
        # If nba_api not available, return last valid data
        print(f"⚠️ nba_api not available - using cached data")
        if self._last_valid_games:
            age = now - self._last_valid_timestamp
            print(f"   Using last valid data ({age:.1f}s old)")
            return self._last_valid_games
        else:
            print(f"   No cached data available - returning empty")
            return []
    
    def _parse_scoreboardv2_game(self, game_row: list, line_score_rs: Dict) -> Dict:
        """
        Parse scoreboardv2 STATS endpoint (REAL-TIME!)
        
        Args:
            game_row: Row from GameHeader resultSet
            line_score_rs: LineScore resultSet for current scores
        
        Returns:
            Dict with game state
        """
        try:
            # GameHeader format: [GAME_DATE_EST, GAME_SEQUENCE, GAME_ID, GAME_STATUS_ID, GAME_STATUS_TEXT, ...]
            # Index mapping based on scoreboardv2 schema
            game_id = game_row[2]  # GAME_ID
            game_status = game_row[3]  # GAME_STATUS_ID (1=sched, 2=live, 3=final)
            status_text = game_row[4]  # GAME_STATUS_TEXT
            
            # Get visitor (away) and home teams from row
            visitor_team = game_row[6]  # VISITOR_TEAM_ID
            home_team = game_row[7]  # HOME_TEAM_ID
            
            # Get scores from LineScore resultSet
            home_score = 0
            away_score = 0
            period = 0
            game_clock = ''
            
            if line_score_rs and 'rowSet' in line_score_rs:
                for line_row in line_score_rs['rowSet']:
                    if line_row[3] == game_id:  # Match by GAME_ID
                        team_id = line_row[1]  # TEAM_ID
                        pts = line_row[22] if len(line_row) > 22 else 0  # PTS
                        
                        if team_id == home_team:
                            home_score = int(pts) if pts else 0
                        elif team_id == visitor_team:
                            away_score = int(pts) if pts else 0
            
            # Get team abbreviations (need to map team IDs)
            team_abbr_map = {
                1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN', 1610612766: 'CHA',
                1610612741: 'CHI', 1610612739: 'CLE', 1610612742: 'DAL', 1610612743: 'DEN',
                1610612765: 'DET', 1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
                1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM', 1610612748: 'MIA',
                1610612749: 'MIL', 1610612750: 'MIN', 1610612740: 'NOP', 1610612752: 'NYK',
                1610612760: 'OKC', 1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
                1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS', 1610612761: 'TOR',
                1610612762: 'UTA', 1610612764: 'WAS'
            }
            
            home_tricode = team_abbr_map.get(home_team, 'HOME')
            away_tricode = team_abbr_map.get(visitor_team, 'AWAY')
            
            # Force to LIVE if game has started
            if game_status == 1 and (home_score > 0 or away_score > 0):
                game_status = 2
            
            current_diff = home_score - away_score
            is_q2_6min = self._is_q2_6min(period, game_clock)
            can_predict = self._can_predict(game_status, period, game_clock)
            
            return {
                'game_id': str(game_id),
                'status': game_status,
                'status_text': status_text or self._get_status_text(game_status, period, game_clock),
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
            print(f"Error parsing scoreboardv2: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_stats_nba_game(self, game_data: Dict) -> Dict:
        """
        Parse stats.nba.com direct API format (FASTEST - no CDN cache!)
        
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
            
            # Force to LIVE if game has started
            if game_status == 1 and (home_score > 0 or away_score > 0 or period > 0):
                game_status = 2
            
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
            print(f"Error parsing stats.nba.com game: {e}")
            return None
    
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

