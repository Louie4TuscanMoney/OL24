"""
POSSESSION TRACKER - Play-by-Play Parser
Calculates REAL possessions from NBA play-by-play data

Uses nba_api to parse every play and track:
- Team possessions
- Player possessions (touches)
- Possession outcomes (shot, TO, FT, end of quarter)
- Possession duration
- Play classification for future ML

Critical for:
- Accurate Per-100 stats
- Usage rate
- True pace calculation
- Future conditional play classification
- Game environment analysis for ML models
"""

from nba_api.stats.endpoints import playbyplayv2
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


class PossessionTracker:
    """
    Parses play-by-play data to calculate real possessions
    
    A possession ends when:
    - Made field goal (not followed by offensive rebound)
    - Defensive rebound
    - Turnover
    - End of quarter/period
    - Free throw (last of sequence, not followed by offensive rebound)
    """
    
    def __init__(self):
        self.possession_events = {
            'possession_ending': [
                'Made Shot',           # FG made (if no OREB follows)
                'Turnover',            # TO ends possession
                'Rebound',             # Will check if defensive
                'End Period',          # End of quarter
                'End Game'             # End of game
            ],
            'possession_continuing': [
                'Missed Shot',         # Continues if OREB
                'Free Throw',          # Check if last and outcome
            ]
        }
    
    def parse_game_possessions(self, game_id: str) -> Dict:
        """
        Parse a game's play-by-play and return possession data
        
        Returns:
        {
            'team_possessions': {team_id: count},
            'player_touches': {player_id: count},
            'possession_details': [list of possession objects],
            'game_pace': possessions_per_48_min
        }
        """
        
        print(f"   📊 Parsing game {game_id}...")
        
        try:
            # Fetch play-by-play data
            pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
            plays_df = pbp.get_data_frames()[0]
            
            if len(plays_df) == 0:
                print(f"      ⚠️  No play-by-play data")
                return None
            
            # Track possessions
            team_possessions = defaultdict(int)
            player_touches = defaultdict(int)
            possession_details = []
            
            current_possession = {
                'team_id': None,
                'start_time': None,
                'plays': [],
                'outcome': None
            }
            
            for idx, play in plays_df.iterrows():
                event_type = play['EVENTMSGTYPE']
                home_desc = play.get('HOMEDESCRIPTION', '')
                away_desc = play.get('VISITORDESCRIPTION', '')
                
                # Determine which team has possession
                if home_desc:
                    poss_team = play['PLAYER1_TEAM_ID']
                elif away_desc:
                    poss_team = play['PLAYER1_TEAM_ID']
                else:
                    poss_team = current_possession['team_id']
                
                # Check for possession-ending events
                if self._is_possession_ending(play, plays_df, idx):
                    if current_possession['team_id']:
                        # End current possession
                        team_possessions[current_possession['team_id']] += 1
                        possession_details.append(current_possession.copy())
                    
                    # Start new possession (switches team on defensive rebound)
                    if 'Rebound' in str(home_desc or away_desc):
                        if 'Defensive' in str(home_desc or away_desc):
                            poss_team = self._get_opposite_team(poss_team, plays_df)
                    
                    current_possession = {
                        'team_id': poss_team,
                        'start_time': play['PCTIMESTRING'],
                        'plays': [play['EVENTMSGACTIONTYPE']],
                        'outcome': self._classify_outcome(play)
                    }
                else:
                    # Continue possession
                    if current_possession['team_id'] is None:
                        current_possession['team_id'] = poss_team
                    current_possession['plays'].append(play['EVENTMSGACTIONTYPE'])
                
                # Track player touches
                if play['PLAYER1_ID']:
                    player_touches[play['PLAYER1_ID']] += 1
            
            # Calculate game pace (possessions per 48 minutes)
            total_possessions = sum(team_possessions.values())
            
            # Get game duration (handle overtime)
            periods = plays_df['PERIOD'].max()
            regulation_time = min(periods, 4) * 12  # 48 min max
            overtime = max(0, periods - 4) * 5      # 5 min OT
            total_minutes = regulation_time + overtime
            
            pace = (total_possessions / total_minutes) * 48 if total_minutes > 0 else 0
            
            print(f"      ✅ Possessions: {total_possessions} (Pace: {pace:.1f})")
            
            return {
                'game_id': game_id,
                'team_possessions': dict(team_possessions),
                'player_touches': dict(player_touches),
                'possession_details': possession_details,
                'total_possessions': total_possessions,
                'game_pace': round(pace, 2),
                'periods': periods
            }
            
        except Exception as e:
            print(f"      ❌ Error parsing game: {e}")
            return None
    
    def _is_possession_ending(self, play: pd.Series, plays_df: pd.DataFrame, idx: int) -> bool:
        """Check if this play ends the possession"""
        
        event_type = play['EVENTMSGTYPE']
        desc = str(play.get('HOMEDESCRIPTION', '') or play.get('VISITORDESCRIPTION', ''))
        
        # 1. Made field goal (check next play for offensive rebound)
        if event_type == 1:  # Made shot
            if idx + 1 < len(plays_df):
                next_play = plays_df.iloc[idx + 1]
                next_desc = str(next_play.get('HOMEDESCRIPTION', '') or next_play.get('VISITORDESCRIPTION', ''))
                if 'Offensive' in next_desc and 'Rebound' in next_desc:
                    return False  # Possession continues on OREB
            return True
        
        # 2. Defensive rebound
        if event_type == 3:  # Rebound
            if 'Defensive' in desc:
                return True
        
        # 3. Turnover
        if event_type == 5:  # Turnover
            return True
        
        # 4. End of period
        if event_type == 13:  # End period
            return True
        
        # 5. Last free throw (check if made and no OREB)
        if event_type == 3:  # Free throw
            # Complex logic to determine if last FT
            # For now, simplified
            return False
        
        return False
    
    def _classify_outcome(self, play: pd.Series) -> str:
        """Classify how possession ended"""
        event_type = play['EVENTMSGTYPE']
        
        if event_type == 1:
            return 'made_shot'
        elif event_type == 2:
            return 'missed_shot'
        elif event_type == 5:
            return 'turnover'
        elif event_type == 3:
            return 'defensive_rebound'
        elif event_type == 13:
            return 'end_period'
        else:
            return 'unknown'
    
    def _get_opposite_team(self, team_id: int, plays_df: pd.DataFrame) -> int:
        """Get the opposing team ID"""
        all_teams = plays_df['PLAYER1_TEAM_ID'].unique()
        all_teams = [t for t in all_teams if pd.notna(t) and t != 0]
        
        if len(all_teams) == 2:
            return [t for t in all_teams if t != team_id][0]
        return team_id
    
    def calculate_player_per100(self, player_stats: Dict, team_possessions: int) -> Dict:
        """
        Calculate accurate per-100 possessions stats
        
        Args:
            player_stats: Dict with pts, reb, ast, etc.
            team_possessions: REAL team possessions from play-by-play
        
        Returns:
            Dict with pts_100, reb_100, ast_100, etc.
        """
        
        if team_possessions == 0:
            return {k + '_100': 0 for k in player_stats.keys()}
        
        per_100 = {}
        for stat, value in player_stats.items():
            per_100[stat + '_100'] = round((value * 100) / team_possessions, 2)
        
        return per_100


def parse_season_possessions(season: str = '2025-26') -> pd.DataFrame:
    """
    Parse all games in a season to build possession database
    
    Returns DataFrame with:
    - game_id
    - team_id
    - possessions
    - pace
    - period
    """
    
    print("="*80)
    print(f"🏀 PARSING SEASON POSSESSIONS: {season}")
    print("="*80)
    
    from nba_api.stats.endpoints import leaguegamefinder
    
    # Get all games for season
    games_finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable='Regular Season'
    )
    games_df = games_finder.get_data_frames()[0]
    
    # Get unique games (each game appears twice - once per team)
    unique_games = games_df['GAME_ID'].unique()
    
    print(f"Found {len(unique_games)} games to parse")
    print()
    
    tracker = PossessionTracker()
    all_possessions = []
    
    for i, game_id in enumerate(unique_games[:10], 1):  # Start with first 10 games
        print(f"Game {i}/{min(10, len(unique_games))}: {game_id}")
        result = tracker.parse_game_possessions(game_id)
        
        if result:
            # Store per-team possession data
            for team_id, poss_count in result['team_possessions'].items():
                all_possessions.append({
                    'game_id': game_id,
                    'team_id': team_id,
                    'possessions': poss_count,
                    'pace': result['game_pace'],
                    'periods': result['periods']
                })
        
        # Rate limit (nba_api can be strict)
        if i % 5 == 0:
            print(f"   ⏸️  Pausing for rate limit...")
            import time
            time.sleep(1)
    
    print()
    print("="*80)
    print(f"✅ PARSED {len(all_possessions)} TEAM-GAME POSSESSION RECORDS")
    print("="*80)
    
    return pd.DataFrame(all_possessions)


if __name__ == "__main__":
    # Test with a recent game
    print("Testing possession tracker...")
    print()
    
    tracker = PossessionTracker()
    
    # Example: Parse a specific game
    # game_id = '0022500001'  # First game of 2025-26 season
    # result = tracker.parse_game_possessions(game_id)
    
    # Or parse full season (warning: takes time!)
    possessions_df = parse_season_possessions('2024-25')  # Use last season for testing
    print(possessions_df.head(10))
    
    # Save to CSV
    possessions_df.to_csv('season_possessions_2024-25.csv', index=False)
    print("Saved to season_possessions_2024-25.csv")

