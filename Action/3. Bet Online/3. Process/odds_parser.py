"""
Odds Parser - Convert raw HTML to structured data
"""

import re
from typing import Dict, Optional

class OddsParser:
    """
    Parse BetOnline odds strings
    """
    
    @staticmethod
    def parse_spread(spread_str: str) -> Optional[float]:
        """
        Parse spread string to float
        
        Examples:
            "+7.5 (-110)" → 7.5
            "-7.5 (-110)" → -7.5
            "PK (-110)" → 0.0
        """
        if not spread_str:
            return None
        
        # Extract number
        match = re.search(r'([+-]?\d+\.?\d*)', spread_str)
        if match:
            return float(match.group(1))
        
        # PK (pick'em) = 0
        if 'PK' in spread_str.upper():
            return 0.0
        
        return None
    
    @staticmethod
    def parse_odds(odds_str: str) -> Optional[int]:
        """
        Parse American odds to integer
        
        Examples:
            "(-110)" → -110
            "(+150)" → 150
        """
        if not odds_str:
            return None
        
        match = re.search(r'\(([+-]?\d+)\)', odds_str)
        if match:
            return int(match.group(1))
        
        return None
    
    @staticmethod
    def implied_probability(american_odds: int) -> float:
        """
        Convert American odds to implied probability
        
        Args:
            american_odds: -110, +150, etc.
        
        Returns:
            Implied probability (0-1)
        """
        if american_odds < 0:
            # Favorite: prob = |odds| / (|odds| + 100)
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            # Underdog: prob = 100 / (odds + 100)
            return 100 / (american_odds + 100)
    
    def normalize_odds(self, raw_odds: Dict) -> Dict:
        """
        Normalize raw odds to standard format
        """
        normalized = {
            'game_id': raw_odds.get('game_id'),
            'away_team': raw_odds.get('away_team', '').upper(),
            'home_team': raw_odds.get('home_team', '').upper(),
            'spread': {
                'away': self.parse_spread(raw_odds.get('spread_away')),
                'home': self.parse_spread(raw_odds.get('spread_home')),
                'away_odds': self.parse_odds(raw_odds.get('spread_away')),
                'home_odds': self.parse_odds(raw_odds.get('spread_home'))
            },
            'timestamp': raw_odds.get('timestamp')
        }
        
        # Calculate implied probabilities
        if normalized['spread']['away_odds']:
            normalized['spread']['away_prob'] = self.implied_probability(
                normalized['spread']['away_odds']
            )
        if normalized['spread']['home_odds']:
            normalized['spread']['home_prob'] = self.implied_probability(
                normalized['spread']['home_odds']
            )
        
        return normalized

