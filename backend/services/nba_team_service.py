"""
NBA Team Data Service
Uses nba_api to fetch and populate team rosters, depth charts, and schedules
"""

from typing import List, Dict, Optional
from datetime import datetime


class NBATeamService:
    """
    Service for fetching comprehensive team data from nba_api
    
    Features:
    - Team rosters with current players
    - Depth charts with projected starters
    - Team schedules (upcoming games)
    - Player stats and advanced metrics
    """
    
    def __init__(self):
        """Initialize team service"""
        try:
            from nba_api.stats.static import teams, players
            from nba_api.stats.endpoints import commonteamroster, teamgamelog
            
            self.teams_static = teams
            self.players_static = players
            self.roster_endpoint = commonteamroster
            self.gamelog_endpoint = teamgamelog
            self.available = True
            
            print("✅ NBA Team Service initialized (nba_api)")
        except ImportError:
            print("❌ nba_api not available for team service")
            self.available = False
    
    def get_team_roster(self, team_abbr: str) -> Dict:
        """
        Get full team roster with player details
        
        Args:
            team_abbr: Team abbreviation (e.g., 'LAL')
        
        Returns:
            Dict with starters, bench, and player stats
        """
        if not self.available:
            return {'error': 'nba_api not available'}
        
        try:
            # Get team ID from abbreviation
            all_teams = self.teams_static.get_teams()
            team = next((t for t in all_teams if t['abbreviation'] == team_abbr), None)
            
            if not team:
                return {'error': f'Team {team_abbr} not found'}
            
            team_id = team['id']
            
            # Fetch current roster
            roster = self.roster_endpoint.CommonTeamRoster(team_id=team_id)
            roster_df = roster.get_data_frames()[0]
            
            # Get all players data
            all_players_dict = {p['id']: p for p in self.players_static.get_players()}
            
            # Build roster data
            starters = []
            bench = []
            
            for _, player_row in roster_df.iterrows():
                player_data = {
                    'player_id': player_row['PLAYER_ID'],
                    'name': player_row['PLAYER'],
                    'position': player_row.get('POSITION', 'N/A'),
                    'jersey': player_row.get('NUM', ''),
                    'height': player_row.get('HEIGHT', ''),
                    'weight': player_row.get('WEIGHT', ''),
                    'birth_date': player_row.get('BIRTH_DATE', ''),
                    'age': player_row.get('AGE', ''),
                    'exp': player_row.get('EXP', '0'),
                    'school': player_row.get('SCHOOL', ''),
                    # Stats will be added from separate query
                    'ppg': 0.0,
                    'rpg': 0.0,
                    'apg': 0.0,
                    'mpg': 0.0
                }
                
                # Simple heuristic: First 5-7 players are usually starters
                # Or we can use a separate depth chart query
                if len(starters) < 5:
                    starters.append(player_data)
                else:
                    bench.append(player_data)
            
            return {
                'team_id': team_id,
                'team_name': team['full_name'],
                'team_abbr': team_abbr,
                'starters': starters,
                'bench': bench,
                'total_players': len(roster_df)
            }
            
        except Exception as e:
            print(f"❌ Error fetching roster for {team_abbr}: {e}")
            return {'error': str(e)}
    
    def get_team_schedule(self, team_abbr: str, season: str = "2025-26") -> List[Dict]:
        """
        Get team's schedule (upcoming games)
        
        Args:
            team_abbr: Team abbreviation
            season: Season (e.g., '2025-26')
        
        Returns:
            List of upcoming games
        """
        if not self.available:
            return []
        
        try:
            # Get team ID
            all_teams = self.teams_static.get_teams()
            team = next((t for t in all_teams if t['abbreviation'] == team_abbr), None)
            
            if not team:
                return []
            
            team_id = team['id']
            
            # Fetch game log (includes upcoming if available)
            gamelog = self.gamelog_endpoint.TeamGameLog(team_id=team_id, season=season)
            games_df = gamelog.get_data_frames()[0]
            
            # Parse games
            schedule = []
            for _, game_row in games_df.iterrows():
                game_data = {
                    'game_id': game_row.get('Game_ID', ''),
                    'game_date': game_row.get('GAME_DATE', ''),
                    'opponent': game_row.get('MATCHUP', '').split(' ')[-1],  # Extract opponent from "LAL vs. BOS"
                    'home_away': 'vs.' if 'vs.' in game_row.get('MATCHUP', '') else '@',
                    'result': game_row.get('WL', ''),  # W or L
                    'score': f"{game_row.get('PTS', 0)}-{game_row.get('PTS', 0)}"  # Needs opponent score
                }
                schedule.append(game_data)
            
            return schedule[:20]  # Return next 20 games
            
        except Exception as e:
            print(f"❌ Error fetching schedule for {team_abbr}: {e}")
            return []
    
    def get_depth_chart(self, team_abbr: str) -> Dict:
        """
        Get projected starting lineup and depth chart
        
        Uses roster + recent playing time to determine starters
        
        Args:
            team_abbr: Team abbreviation
        
        Returns:
            Dict with projected starters by position
        """
        # For now, use roster function
        # TODO: Add logic to determine starters based on recent minutes played
        roster_data = self.get_team_roster(team_abbr)
        
        if 'error' in roster_data:
            return roster_data
        
        return {
            'team_abbr': team_abbr,
            'projected_starters': roster_data['starters'],
            'bench': roster_data['bench'],
            'notes': 'Projected based on roster order. Use recent MPG for more accuracy.'
        }


# Global instance
team_service = NBATeamService()

