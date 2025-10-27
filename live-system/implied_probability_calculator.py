"""
IMPLIED PROBABILITY CALCULATOR
Converts American odds to implied probabilities in real-time
"""

from typing import Dict, Tuple


class ImpliedProbabilityCalculator:
    """
    Calculate implied probabilities from American odds
    ELON SPEED: <0.001s per calculation
    """
    
    @staticmethod
    def american_to_probability(american_odds: int) -> float:
        """
        Convert American odds to implied probability
        
        Args:
            american_odds: American odds (e.g., -150, +130)
            
        Returns:
            Implied probability (0-1)
        """
        if american_odds > 0:
            # Underdog (e.g., +130)
            return 100 / (american_odds + 100)
        else:
            # Favorite (e.g., -150)
            return abs(american_odds) / (abs(american_odds) + 100)
    
    @staticmethod
    def calculate_no_vig_probability(
        favorite_odds: int,
        underdog_odds: int
    ) -> Tuple[float, float]:
        """
        Remove vig (bookmaker edge) to get true probabilities
        
        Args:
            favorite_odds: Favorite's American odds (negative)
            underdog_odds: Underdog's American odds (positive)
            
        Returns:
            (favorite_prob, underdog_prob) without vig
        """
        # Get implied probabilities
        fav_implied = ImpliedProbabilityCalculator.american_to_probability(favorite_odds)
        dog_implied = ImpliedProbabilityCalculator.american_to_probability(underdog_odds)
        
        # Total is > 1.0 due to vig
        total = fav_implied + dog_implied
        
        # Remove vig by normalizing
        fav_no_vig = fav_implied / total
        dog_no_vig = dog_implied / total
        
        return fav_no_vig, dog_no_vig
    
    @staticmethod
    def calculate_spread_probabilities(
        home_ml: int,
        away_ml: int,
        spread: float
    ) -> Dict:
        """
        Calculate all probabilities for a spread bet
        
        Args:
            home_ml: Home team moneyline
            away_ml: Away team moneyline
            spread: Point spread
            
        Returns:
            Dict with all probabilities
        """
        # Implied probabilities (with vig)
        home_implied = ImpliedProbabilityCalculator.american_to_probability(home_ml)
        away_implied = ImpliedProbabilityCalculator.american_to_probability(away_ml)
        
        # No-vig probabilities (true market)
        if home_ml < 0 and away_ml > 0:
            home_no_vig, away_no_vig = ImpliedProbabilityCalculator.calculate_no_vig_probability(
                home_ml, away_ml
            )
        elif away_ml < 0 and home_ml > 0:
            away_no_vig, home_no_vig = ImpliedProbabilityCalculator.calculate_no_vig_probability(
                away_ml, home_ml
            )
        else:
            # Both same sign (rare, just use implied)
            total = home_implied + away_implied
            home_no_vig = home_implied / total
            away_no_vig = away_implied / total
        
        # Bookmaker edge (vig)
        vig = (home_implied + away_implied - 1.0) * 100
        
        return {
            'home_implied': round(home_implied, 4),
            'away_implied': round(away_implied, 4),
            'home_no_vig': round(home_no_vig, 4),
            'away_no_vig': round(away_no_vig, 4),
            'vig_percentage': round(vig, 2),
            'spread': spread,
            'home_ml': home_ml,
            'away_ml': away_ml
        }


# Test
if __name__ == "__main__":
    calc = ImpliedProbabilityCalculator()
    
    print("="*80)
    print("IMPLIED PROBABILITY CALCULATOR TEST")
    print("="*80)
    print()
    
    # Test case: Pick'em (0.0 spread)
    test_cases = [
        {"name": "Pick'em", "home_ml": -110, "away_ml": -110, "spread": 0.0},
        {"name": "Small favorite", "home_ml": -150, "away_ml": +130, "spread": -3.0},
        {"name": "Big favorite", "home_ml": -300, "away_ml": +250, "spread": -7.5},
    ]
    
    for test in test_cases:
        print(f"📊 {test['name']}: Spread {test['spread']}")
        probs = calc.calculate_spread_probabilities(
            test['home_ml'], test['away_ml'], test['spread']
        )
        print(f"   Home ML: {test['home_ml']:+4d} → Implied: {probs['home_implied']:.1%}, No-vig: {probs['home_no_vig']:.1%}")
        print(f"   Away ML: {test['away_ml']:+4d} → Implied: {probs['away_implied']:.1%}, No-vig: {probs['away_no_vig']:.1%}")
        print(f"   Vig: {probs['vig_percentage']:.2f}%")
        print()

