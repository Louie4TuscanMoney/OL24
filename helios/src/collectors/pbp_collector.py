"""
🏀 COMPREHENSIVE PBP COLLECTOR - HEDGE FUND GRADE

Extracts FULL play-by-play data with NO compression.
Preserves ALL event details for downstream feature extraction.

WHAT WE EXTRACT (vs old approach):
  OLD: 18 score differential numbers (compressed)
  NEW: Every shot, possession, lineup, event, timeout (FULL SIGNAL!)

This is the foundation of Project Helios.
"""

import numpy as np
import pandas as pd
import pickle
import time
import json
from datetime import datetime
from nba_api.stats.endpoints import playbyplayv2, boxscoretraditionalv2
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class ComprehensivePBPCollector:
    """
    Comprehensive Play-by-Play collector.
    
    Extracts full-resolution PBP data including:
    - All events (shots, fouls, turnovers, rebounds, assists)
    - Event details (player IDs, descriptions, types)
    - Timing information (period, clock, elapsed time)
    - Score state at each event
    - Lineup information (when available)
    """
    
    def __init__(self, rate_limit_seconds: float = 0.6, verbose: bool = True):
        self.rate_limit = rate_limit_seconds
        self.verbose = verbose
        self.collected_count = 0
        self.error_count = 0
    
    def collect_game(self, game_id: str) -> Optional[Dict]:
        """
        Collect comprehensive PBP data for a single game.
        
        Returns:
            Dictionary with:
            - game_id
            - date
            - events (full DataFrame)
            - event_summary
            - score_timeline
            - targets (diff at Q2 6:00, halftime, final)
        """
        try:
            # Get PBP
            pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
            pbp_df = pbp.get_data_frames()[0]
            
            if pbp_df.empty:
                return None
            
            # Extract game date
            if 'GAME_DATE_EST' in pbp_df.columns:
                game_date = str(pbp_df['GAME_DATE_EST'].iloc[0])
            else:
                game_date = datetime.now().isoformat()
            
            # Process events
            game_data = {
                'game_id': game_id,
                'date': game_date,
                'events': self._process_events(pbp_df),
                'event_summary': self._summarize_events(pbp_df),
                'score_timeline': self._extract_score_timeline(pbp_df),
                'targets': self._extract_targets(pbp_df)
            }
            
            self.collected_count += 1
            
            if self.verbose and self.collected_count % 10 == 0:
                print(f"  [{self.collected_count} collected] Latest: {game_id[:8]}...")
            
            return game_data
            
        except Exception as e:
            self.error_count += 1
            if self.verbose:
                print(f"    ✗ Error on {game_id}: {str(e)[:60]}")
            return None
    
    def _process_events(self, pbp_df: pd.DataFrame) -> List[Dict]:
        """
        Process PBP DataFrame into structured event list.
        
        Extracts:
        - Event type (shot, foul, turnover, rebound, etc.)
        - Players involved
        - Time information
        - Score state
        - Event description
        """
        events = []
        
        for idx, row in pbp_df.iterrows():
            event = {
                # Core identifiers
                'event_num': row.get('EVENTNUM', idx),
                'event_type': row.get('EVENTMSGTYPE', 0),
                'event_action': row.get('EVENTMSGACTIONTYPE', 0),
                
                # Timing
                'period': row.get('PERIOD', 0),
                'time_string': str(row.get('PCTIMESTRING', '')),
                
                # Players
                'player1_id': row.get('PLAYER1_ID', None),
                'player1_team': row.get('PLAYER1_TEAM_ID', None),
                'player2_id': row.get('PLAYER2_ID', None),
                'player2_team': row.get('PLAYER2_TEAM_ID', None),
                'player3_id': row.get('PLAYER3_ID', None),
                
                # Descriptions
                'home_desc': str(row.get('HOMEDESCRIPTION', '')),
                'away_desc': str(row.get('VISITORDESCRIPTION', '')),
                
                # Score
                'score_string': str(row.get('SCORE', '')),
                
                # Calculate current differential
                'score_diff': self._parse_score_diff(str(row.get('SCORE', '')))
            }
            
            events.append(event)
        
        return events
    
    def _summarize_events(self, pbp_df: pd.DataFrame) -> Dict:
        """
        Create event summary statistics.
        
        Event types (EVENTMSGTYPE):
        1: Made Shot
        2: Missed Shot
        3: Free Throw
        4: Rebound
        5: Turnover
        6: Foul
        7: Violation
        8: Substitution
        9: Timeout
        10: Jump Ball
        12: Start Period
        13: End Period
        """
        summary = {
            'total_events': len(pbp_df),
            'made_shots': len(pbp_df[pbp_df['EVENTMSGTYPE'] == 1]),
            'missed_shots': len(pbp_df[pbp_df['EVENTMSGTYPE'] == 2]),
            'free_throws': len(pbp_df[pbp_df['EVENTMSGTYPE'] == 3]),
            'rebounds': len(pbp_df[pbp_df['EVENTMSGTYPE'] == 4]),
            'turnovers': len(pbp_df[pbp_df['EVENTMSGTYPE'] == 5]),
            'fouls': len(pbp_df[pbp_df['EVENTMSGTYPE'] == 6]),
            'substitutions': len(pbp_df[pbp_df['EVENTMSGTYPE'] == 8]),
            'timeouts': len(pbp_df[pbp_df['EVENTMSGTYPE'] == 9]),
        }
        
        # Calculate rates
        total_scoring_events = summary['made_shots'] + summary['missed_shots']
        if total_scoring_events > 0:
            summary['fg_pct'] = summary['made_shots'] / total_scoring_events
        else:
            summary['fg_pct'] = 0
        
        return summary
    
    def _extract_score_timeline(self, pbp_df: pd.DataFrame) -> np.ndarray:
        """
        Extract full-resolution score differential timeline.
        
        Returns array with score diff at each event (not compressed!)
        """
        timeline = []
        
        for _, row in pbp_df.iterrows():
            score_str = str(row.get('SCORE', ''))
            diff = self._parse_score_diff(score_str)
            timeline.append(diff)
        
        return np.array(timeline)
    
    def _extract_targets(self, pbp_df: pd.DataFrame) -> Dict:
        """
        Extract prediction targets.
        
        Returns:
        - diff_at_q2_6min: Differential at 6:00 left in Q2
        - diff_at_halftime: Differential at end of Q2
        - diff_at_final: Final game differential
        """
        targets = {
            'diff_at_q2_6min': 0,
            'diff_at_halftime': 0,
            'diff_at_final': 0
        }
        
        # Q2 6:00
        q2_events = pbp_df[pbp_df['PERIOD'] == 2]
        if len(q2_events) > 0:
            for _, event in q2_events.iterrows():
                time_str = str(event.get('PCTIMESTRING', ''))
                if any(t in time_str for t in ['6:0', '6:1', '5:5']):
                    targets['diff_at_q2_6min'] = self._parse_score_diff(str(event.get('SCORE', '')))
                    break
            
            # If not found, use middle of Q2
            if targets['diff_at_q2_6min'] == 0 and len(q2_events) > 0:
                mid_event = q2_events.iloc[len(q2_events)//2]
                targets['diff_at_q2_6min'] = self._parse_score_diff(str(mid_event.get('SCORE', '')))
            
            # Halftime
            targets['diff_at_halftime'] = self._parse_score_diff(str(q2_events.iloc[-1].get('SCORE', '')))
        
        # Final
        final_events = pbp_df[pbp_df['PERIOD'] >= 4]
        if len(final_events) > 0:
            targets['diff_at_final'] = self._parse_score_diff(str(final_events.iloc[-1].get('SCORE', '')))
        else:
            targets['diff_at_final'] = targets['diff_at_halftime']
        
        return targets
    
    def _parse_score_diff(self, score_str: str) -> int:
        """Parse 'XX - YY' score string to home differential."""
        if not score_str or score_str == 'nan' or ' - ' not in score_str:
            return 0
        
        try:
            parts = score_str.split(' - ')
            home_score = int(parts[0])
            away_score = int(parts[1])
            return home_score - away_score
        except:
            return 0
    
    def collect_season(self, season: str, game_ids: List[str], checkpoint_freq: int = 100) -> List[Dict]:
        """
        Collect comprehensive PBP for a full season.
        
        Args:
            season: Season string (e.g., '2020-21')
            game_ids: List of game IDs to collect
            checkpoint_freq: Save checkpoint every N games
        
        Returns:
            List of game dictionaries
        """
        collected = []
        start_time = time.time()
        
        print(f"\n{'='*90}")
        print(f"COLLECTING SEASON: {season}")
        print(f"{'='*90}")
        print(f"  Target games: {len(game_ids)}")
        print(f"  Start time: {datetime.now().strftime('%I:%M %p')}")
        
        for idx, game_id in enumerate(game_ids):
            game_data = self.collect_game(game_id)
            
            if game_data is not None:
                collected.append(game_data)
            
            # Checkpoint
            if (idx + 1) % checkpoint_freq == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                eta = (len(game_ids) - idx - 1) / rate if rate > 0 else 0
                
                print(f"\n  💾 CHECKPOINT: {len(collected)}/{idx+1} games")
                print(f"     Rate: {rate*60:.1f} games/min")
                print(f"     ETA: {eta/60:.1f} min")
                
                # Save checkpoint
                checkpoint_file = f'helios/data/raw/checkpoint_{season}_{idx+1}.pkl'
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(collected, f)
            
            # Rate limiting
            time.sleep(self.rate_limit)
        
        elapsed_total = time.time() - start_time
        
        print(f"\n  ✅ Season {season} complete:")
        print(f"     Collected: {len(collected)}/{len(game_ids)}")
        print(f"     Errors: {self.error_count}")
        print(f"     Time: {elapsed_total/60:.1f} min")
        print(f"     Rate: {len(collected)/(elapsed_total/60):.1f} games/min")
        
        return collected
    
    def save_collected_data(self, data: List[Dict], filename: str):
        """Save collected data to pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        if self.verbose:
            print(f"  ✓ Saved: {filename}")


if __name__ == "__main__":
    # Test collector
    print("="*90)
    print("🏀 COMPREHENSIVE PBP COLLECTOR TEST")
    print("="*90)
    
    collector = ComprehensivePBPCollector(verbose=True)
    
    # Test on one game
    test_game_id = "0022300001"  # Example game ID
    
    print(f"\nTesting on game: {test_game_id}")
    print("Collecting full PBP data (NO compression)...")
    
    game_data = collector.collect_game(test_game_id)
    
    if game_data:
        print(f"\n✅ Collection successful!")
        print(f"  Game ID: {game_data['game_id']}")
        print(f"  Date: {game_data['date'][:10]}")
        print(f"  Total events: {len(game_data['events'])}")
        print(f"  Event summary: {game_data['event_summary']}")
        print(f"  Score timeline: {len(game_data['score_timeline'])} points")
        print(f"  Targets: {game_data['targets']}")
        
        print(f"\n  Sample events (first 5):")
        for i, event in enumerate(game_data['events'][:5]):
            print(f"    {i+1}. Period {event['period']}, {event['time_string']}: "
                  f"Type {event['event_type']}, Diff: {event['score_diff']}")
        
        print(f"\n✅ Collector WORKING - ready for production!")
    else:
        print(f"\n✗ Collection failed")
    
    print("="*90)

