"""
3D BASKETBALL COURT STREAM

Purpose: Generate 3D player positions from play-by-play data for Three.js visualization
Author: Ontologic XYZ
Date: October 20, 2025

This creates a stream of player positions for 3D court visualization
"""

import numpy as np
from typing import Dict, List, Tuple
import json


class Court3DStream:
    """
    Generate 3D player position stream from PBP data
    """
    
    # NBA court dimensions (feet)
    COURT_LENGTH = 94.0
    COURT_WIDTH = 50.0
    
    # Player positions (approximate zones)
    POSITIONS = {
        'PG': (0, 0),     # Point guard (top of key)
        'SG': (1, 0),     # Shooting guard (wing)
        'SF': (-1, 0),    # Small forward (opposite wing)
        'PF': (0.5, -1),  # Power forward (post)
        'C': (0, -1.5)    # Center (paint)
    }
    
    def __init__(self):
        """Initialize 3D court stream generator"""
        pass
    
    def generate_player_positions(
        self,
        event: Dict,
        home_team: str,
        away_team: str
    ) -> Dict:
        """
        Generate 3D player positions for an event
        
        Args:
            event: PBP event dict
            home_team: Home team code
            away_team: Away team code
            
        Returns:
            Dict with player positions for both teams
        """
        event_type = event.get('EVENTMSGTYPE', 0)
        player_id = event.get('PLAYER1_ID')
        
        # Determine which half (offense/defense)
        # This would be more sophisticated with real tracking data
        
        # Generate positions (simplified - in production use real tracking)
        home_players = self._generate_team_positions(
            team=home_team,
            on_offense=True,
            event_type=event_type
        )
        
        away_players = self._generate_team_positions(
            team=away_team,
            on_offense=False,
            event_type=event_type
        )
        
        return {
            'timestamp': event.get('PCTIMESTRING', '12:00'),
            'period': event.get('PERIOD', 1),
            'event_type': self._get_event_name(event_type),
            'home_players': home_players,
            'away_players': away_players,
            'ball_position': self._get_ball_position(event)
        }
    
    def _generate_team_positions(
        self,
        team: str,
        on_offense: bool,
        event_type: int
    ) -> List[Dict]:
        """
        Generate positions for 5 players
        
        Args:
            team: Team code
            on_offense: True if team has ball
            event_type: Type of event
            
        Returns:
            List of player positions
        """
        players = []
        base_positions = list(self.POSITIONS.values())
        
        # Flip coordinates for defense
        if not on_offense:
            base_positions = [(-x, -y) for x, y in base_positions]
        
        # Add some randomness for realism
        for i, (x, y) in enumerate(base_positions):
            # Convert to court coordinates
            court_x = (x * 10) + (self.COURT_LENGTH / 4)
            court_y = (y * 8) + (self.COURT_WIDTH / 2)
            
            # Add noise
            court_x += np.random.uniform(-3, 3)
            court_y += np.random.uniform(-2, 2)
            
            players.append({
                'player_id': f"{team}_{i+1}",
                'x': court_x,
                'y': court_y,
                'z': 0,  # Ground level
                'team': team
            })
        
        return players
    
    def _get_ball_position(self, event: Dict) -> Dict:
        """Get ball position for event"""
        # Simplified - in production would use actual ball tracking
        event_type = event.get('EVENTMSGTYPE', 0)
        
        # Shot event: ball in air
        if event_type == 1:
            return {'x': 47, 'y': 25, 'z': 10}  # In air
        
        # Pass/dribble: ball with player
        return {'x': 40, 'y': 25, 'z': 3}  # With player
    
    def _get_event_name(self, event_type: int) -> str:
        """Convert event type to name"""
        events = {
            1: "SHOT",
            2: "MISS",
            3: "FREE_THROW",
            4: "REBOUND",
            5: "TURNOVER",
            6: "FOUL",
            8: "SUBSTITUTION",
            12: "START_PERIOD",
            13: "END_PERIOD"
        }
        return events.get(event_type, "OTHER")
    
    def generate_game_stream(
        self,
        pbp_events: List[Dict],
        home_team: str,
        away_team: str
    ) -> List[Dict]:
        """
        Generate complete 3D stream for a game
        
        Args:
            pbp_events: List of PBP events
            home_team: Home team code
            away_team: Away team code
            
        Returns:
            List of 3D frames
        """
        frames = []
        
        for event in pbp_events:
            frame = self.generate_player_positions(event, home_team, away_team)
            frames.append(frame)
        
        return frames
    
    def export_for_threejs(
        self,
        frames: List[Dict],
        filepath: str = "court_3d_stream.json"
    ):
        """Export stream in format for Three.js"""
        with open(filepath, 'w') as f:
            json.dump({
                'court_dimensions': {
                    'length': self.COURT_LENGTH,
                    'width': self.COURT_WIDTH
                },
                'frames': frames,
                'fps': 1  # 1 frame per second (PBP resolution)
            }, f, indent=2)
        
        print(f"✅ Exported {len(frames)} frames to {filepath}")


def example_3d_stream():
    """Example: Generate 3D stream"""
    print("\n" + "="*80)
    print("🏀 3D COURT STREAM - EXAMPLE")
    print("="*80 + "\n")
    
    stream = Court3DStream()
    
    # Example PBP events
    example_events = [
        {'EVENTMSGTYPE': 12, 'PERIOD': 1, 'PCTIMESTRING': '12:00'},
        {'EVENTMSGTYPE': 1, 'PERIOD': 1, 'PCTIMESTRING': '11:45', 'PLAYER1_ID': 1},
        {'EVENTMSGTYPE': 2, 'PERIOD': 1, 'PCTIMESTRING': '11:30', 'PLAYER1_ID': 2},
    ]
    
    # Generate stream
    frames = stream.generate_game_stream(
        pbp_events=example_events,
        home_team='LAL',
        away_team='BOS'
    )
    
    print(f"✅ Generated {len(frames)} 3D frames\n")
    
    # Show first frame
    print("Sample frame (players on court):")
    print(json.dumps(frames[0], indent=2)[:500] + "...")
    
    # Export for Three.js
    stream.export_for_threejs(frames)
    
    print("\n" + "="*80)
    print("✅ 3D COURT STREAM READY FOR THREE.JS")
    print("="*80)


if __name__ == "__main__":
    example_3d_stream()

