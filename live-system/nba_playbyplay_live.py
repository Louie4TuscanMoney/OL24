"""
NBA LIVE PLAY-BY-PLAY DATA FETCHER

Purpose: Fetch real-time play-by-play data for Mamba feature extraction
Author: Ontologic XYZ
Date: October 27, 2025

This fetches live PBP data from NBA API for the Mamba 18-minute pattern extraction.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import requests

try:
    from nba_api.live.nba.endpoints import playbyplay
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("⚠️ nba_api not available for play-by-play")


class NBAPlayByPlayLive:
    """
    Fetch live play-by-play data from NBA API
    """
    
    def __init__(self):
        """Initialize play-by-play fetcher"""
        self.cache = {}
        self.cache_duration = 10  # Cache for 10 seconds
        
    def get_live_playbyplay(self, game_id: str) -> Optional[Dict]:
        """
        Get live play-by-play data for a game
        
        Args:
            game_id: NBA game ID (e.g., "0022000180")
            
        Returns:
            Play-by-play data dict or None
        """
        # Check cache
        if game_id in self.cache:
            cached_time, cached_data = self.cache[game_id]
            if time.time() - cached_time < self.cache_duration:
                return cached_data
        
        # Fetch fresh data
        try:
            if NBA_API_AVAILABLE:
                pbp = playbyplay.PlayByPlay(game_id)
                data = pbp.get_dict()
                
                # Cache the result
                self.cache[game_id] = (time.time(), data)
                
                return data
            else:
                print("⚠️ NBA API not available, using fallback")
                return self._fetch_playbyplay_fallback(game_id)
                
        except Exception as e:
            print(f"❌ Error fetching play-by-play for {game_id}: {e}")
            return None
    
    def _fetch_playbyplay_fallback(self, game_id: str) -> Optional[Dict]:
        """
        Fallback method using direct CDN request
        
        Args:
            game_id: NBA game ID
            
        Returns:
            Play-by-play data or None
        """
        try:
            url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ CDN request failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Fallback error: {e}")
            return None
    
    def get_actions_for_period(
        self, 
        game_id: str, 
        period: int, 
        start_clock: str = None,
        end_clock: str = None
    ) -> List[Dict]:
        """
        Get actions for a specific period and time range
        
        Args:
            game_id: NBA game ID
            period: Period number (1-4 for regulation, 5+ for OT)
            start_clock: Start clock (e.g., "PT12M00.00S")
            end_clock: End clock (e.g., "PT6M00.00S")
            
        Returns:
            List of action dictionaries
        """
        pbp_data = self.get_live_playbyplay(game_id)
        
        if not pbp_data or 'game' not in pbp_data:
            return []
        
        actions = pbp_data['game'].get('actions', [])
        
        # Filter by period
        period_actions = [a for a in actions if a.get('period') == period]
        
        # Filter by time range if specified
        if start_clock and end_clock:
            filtered_actions = []
            for action in period_actions:
                clock = action.get('clock', '')
                if self._is_clock_in_range(clock, start_clock, end_clock):
                    filtered_actions.append(action)
            return filtered_actions
        
        return period_actions
    
    def _is_clock_in_range(self, clock: str, start: str, end: str) -> bool:
        """
        Check if a clock time is within a range
        
        Args:
            clock: Clock string (e.g., "PT11M58.00S")
            start: Start clock
            end: End clock
            
        Returns:
            True if in range
        """
        try:
            # Convert PT11M58.00S to seconds
            clock_seconds = self._parse_clock_to_seconds(clock)
            start_seconds = self._parse_clock_to_seconds(start)
            end_seconds = self._parse_clock_to_seconds(end)
            
            # In NBA, clock counts down, so start > end
            return end_seconds <= clock_seconds <= start_seconds
            
        except:
            return False
    
    def _parse_clock_to_seconds(self, clock: str) -> float:
        """
        Parse NBA clock format to seconds
        
        Args:
            clock: Clock string (e.g., "PT11M58.00S")
            
        Returns:
            Seconds as float
        """
        # Remove PT prefix
        clock = clock.replace('PT', '')
        
        # Parse minutes and seconds
        minutes = 0
        seconds = 0
        
        if 'M' in clock:
            parts = clock.split('M')
            minutes = float(parts[0])
            if len(parts) > 1 and 'S' in parts[1]:
                seconds = float(parts[1].replace('S', ''))
        elif 'S' in clock:
            seconds = float(clock.replace('S', ''))
        
        return minutes * 60 + seconds
    
    def get_18min_pattern(self, game_id: str, current_period: int) -> Optional[List[Dict]]:
        """
        Get the last 18 minutes of play-by-play data for Mamba feature extraction
        
        This is the REAL 18-minute pattern that Mamba was trained on!
        
        Args:
            game_id: NBA game ID
            current_period: Current period
            
        Returns:
            List of actions from last 18 minutes
        """
        pbp_data = self.get_live_playbyplay(game_id)
        
        if not pbp_data or 'game' not in pbp_data:
            return None
        
        all_actions = pbp_data['game'].get('actions', [])
        
        # Calculate which periods to include
        # 18 minutes = 1.5 quarters (each quarter is 12 minutes)
        
        if current_period == 1:
            # Not enough data yet
            return None
        elif current_period == 2:
            # Get Q1 (12 min) + first 6 min of Q2
            q1_actions = [a for a in all_actions if a.get('period') == 1]
            q2_actions = [a for a in all_actions if a.get('period') == 2]
            # Filter Q2 to first 6 minutes (12:00 to 6:00)
            q2_first_6min = [
                a for a in q2_actions 
                if self._parse_clock_to_seconds(a.get('clock', 'PT0M0S')) >= 360  # >= 6:00
            ]
            return q1_actions + q2_first_6min
        else:
            # Q3+: Get previous full quarter + first 6 min of current quarter
            prev_period = current_period - 1
            prev_actions = [a for a in all_actions if a.get('period') == prev_period]
            curr_actions = [a for a in all_actions if a.get('period') == current_period]
            # Filter current to first 6 minutes
            curr_first_6min = [
                a for a in curr_actions 
                if self._parse_clock_to_seconds(a.get('clock', 'PT0M0S')) >= 360
            ]
            return prev_actions + curr_first_6min
    
    def extract_scoring_events(self, actions: List[Dict]) -> List[Dict]:
        """
        Extract scoring events from actions
        
        Args:
            actions: List of action dictionaries
            
        Returns:
            List of scoring events
        """
        scoring_types = ['2pt', '3pt', 'freethrow']
        
        scoring_events = []
        for action in actions:
            action_type = action.get('actionType', '')
            if action_type in scoring_types or action.get('isFieldGoal') == 1:
                scoring_events.append({
                    'period': action.get('period'),
                    'clock': action.get('clock'),
                    'teamId': action.get('teamId'),
                    'teamTricode': action.get('teamTricode'),
                    'actionType': action_type,
                    'scoreHome': action.get('scoreHome'),
                    'scoreAway': action.get('scoreAway'),
                    'playerName': action.get('playerName'),
                    'description': action.get('description')
                })
        
        return scoring_events
    
    def calculate_pattern_stats(self, actions: List[Dict]) -> Dict:
        """
        Calculate statistics from play-by-play pattern
        
        This generates the features that Mamba was trained on!
        
        Args:
            actions: List of action dictionaries
            
        Returns:
            Dictionary of pattern statistics
        """
        if not actions:
            return {}
        
        # Extract scoring events
        scoring_events = self.extract_scoring_events(actions)
        
        # Calculate stats
        total_points = len(scoring_events)
        
        # Points by team
        home_points = sum(1 for e in scoring_events if e.get('scoreHome', '0') > e.get('scoreAway', '0'))
        away_points = sum(1 for e in scoring_events if e.get('scoreAway', '0') > e.get('scoreHome', '0'))
        
        # Pace (actions per minute)
        if actions:
            first_clock = self._parse_clock_to_seconds(actions[0].get('clock', 'PT12M0S'))
            last_clock = self._parse_clock_to_seconds(actions[-1].get('clock', 'PT0M0S'))
            time_elapsed = (first_clock - last_clock) / 60.0  # minutes
            pace = len(actions) / max(time_elapsed, 1.0)
        else:
            pace = 0
        
        return {
            'total_actions': len(actions),
            'total_scoring_events': total_points,
            'home_scoring_events': home_points,
            'away_scoring_events': away_points,
            'pace': pace,
            'pattern_length_minutes': 18.0
        }


def test_live_playbyplay():
    """
    Test the live play-by-play fetcher
    """
    print("\n" + "="*80)
    print("🏀 TESTING LIVE PLAY-BY-PLAY FETCHER")
    print("="*80 + "\n")
    
    fetcher = NBAPlayByPlayLive()
    
    # Test with a recent game ID (you'll need to replace with actual live game)
    game_id = "0022000180"  # Example game ID
    
    print(f"📊 Fetching play-by-play for game {game_id}...")
    pbp_data = fetcher.get_live_playbyplay(game_id)
    
    if pbp_data:
        print(f"✅ Successfully fetched play-by-play data")
        print(f"   Total actions: {len(pbp_data.get('game', {}).get('actions', []))}")
        
        # Get 18-minute pattern
        pattern = fetcher.get_18min_pattern(game_id, current_period=2)
        if pattern:
            print(f"\n📈 18-minute pattern extracted:")
            print(f"   Actions in pattern: {len(pattern)}")
            
            # Calculate stats
            stats = fetcher.calculate_pattern_stats(pattern)
            print(f"\n📊 Pattern statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        else:
            print("⚠️ Not enough data for 18-minute pattern")
    else:
        print("❌ Failed to fetch play-by-play data")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    test_live_playbyplay()

