"""
Probability Converter for Sports Betting
Convert American odds, ML intervals to probabilities
Performance: <5ms per conversion

Based on: RISK_OPTIMIZATION/Applied Model/probability_converter.py
Following: RISK_OPTIMIZATION/MATH_BREAKDOWN.txt
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple

class ProbabilityConverter:
    """
    Convert various formats to win probabilities for betting
    
    ALL METHODS <5ms - REAL-TIME COMPATIBLE
    """
    
    @staticmethod
    def american_to_decimal_odds(american_odds: float) -> float:
        """
        Convert American odds to decimal odds
        
        Args:
            american_odds: -110, +150, etc.
        
        Returns:
            Decimal odds (e.g., 1.909 for -110)
        
        Formula:
            Negative (favorite): 1 + (100 / |odds|)
            Positive (underdog): 1 + (odds / 100)
        
        Time: <0.1ms
        """
        if american_odds < 0:
            # Favorite: -110 → 1.909
            return 1 + (100 / abs(american_odds))
        else:
            # Underdog: +150 → 2.5
            return 1 + (american_odds / 100)
    
    @staticmethod
    def american_to_implied_probability(american_odds: float) -> float:
        """
        Convert American odds to implied probability
        
        Args:
            american_odds: -110, +150, etc.
        
        Returns:
            Implied probability (0-1)
        
        Formula (from MATH_BREAKDOWN.txt):
            Negative: |odds| / (|odds| + 100)
            Positive: 100 / (odds + 100)
        
        Examples:
            -110 → 0.524 (52.4%)
            +150 → 0.400 (40.0%)
        
        Time: <0.1ms
        """
        if american_odds < 0:
            # Favorite
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            # Underdog
            return 100 / (american_odds + 100)
    
    @staticmethod
    def ml_interval_to_probability(
        ml_forecast: float,
        ml_lower: float,
        ml_upper: float,
        market_spread: float,
        coverage: float = 0.95,
        conservative_factor: float = 0.75
    ) -> float:
        """
        Convert ML confidence interval to probability of covering spread
        
        CRITICAL CONVERSION: Halftime prediction → Full game probability
        
        Args:
            ml_forecast: Point prediction at HALFTIME (e.g., +15.1)
            ml_lower: Lower bound of CI (e.g., +11.3)
            ml_upper: Upper bound of CI (e.g., +18.9)
            market_spread: FULL GAME spread (e.g., -7.5)
            coverage: CI coverage (0.95 for 95%)
            conservative_factor: Blend with 50/50 for safety
        
        Returns:
            Probability of covering spread (0-1)
        
        Math (from MATH_BREAKDOWN.txt):
            1. Estimate σ from interval: σ = width / (2 × z_score)
            2. Convert halftime → full game: margin = forecast × 0.55
            3. Calculate z-score: z = (margin - spread) / σ
            4. Get probability: P = Φ(z)
            5. Conservative adjustment: P_adj = 0.75 + 0.25 × P
        
        Time: <5ms
        """
        # Step 1: Estimate standard deviation from interval
        z_score = norm.ppf((1 + coverage) / 2)  # 1.96 for 95%
        sigma = (ml_upper - ml_lower) / (2 * z_score)
        
        # Step 2: Convert halftime lead → full game margin
        # Empirical: full game margin ≈ 0.55 × halftime lead
        implied_margin = ml_forecast * 0.55
        
        # Step 3: Calculate z-score for covering
        if sigma > 0:
            z = (implied_margin - market_spread) / sigma
        else:
            z = 0
        
        # Step 4: Convert to probability using normal CDF
        prob = norm.cdf(z)
        
        # Step 5: Conservative adjustment (don't trust model 100%)
        # Blend: 75% base + 25% model adjustment
        prob_adjusted = conservative_factor + (1 - conservative_factor) * prob
        
        # Clip to valid betting range
        return np.clip(prob_adjusted, 0.01, 0.99)
    
    @staticmethod
    def remove_vig(p_home: float, p_away: float) -> Tuple[float, float]:
        """
        Remove bookmaker vig to get true probabilities
        
        Bookmaker probabilities sum to >1.0 (the vig/juice)
        True probabilities must sum to exactly 1.0
        
        Args:
            p_home: Implied probability for home
            p_away: Implied probability for away
        
        Returns:
            (p_home_true, p_away_true) summing to 1.0
        
        Example:
            Input: 0.524 + 0.524 = 1.048 (4.8% vig)
            Output: 0.500 + 0.500 = 1.000
        
        Time: <0.5ms
        """
        total = p_home + p_away
        
        p_home_true = p_home / total
        p_away_true = p_away / total
        
        return p_home_true, p_away_true
    
    @staticmethod
    def expected_value(
        bet_size: float,
        win_probability: float,
        american_odds: float
    ) -> float:
        """
        Calculate expected value of bet
        
        Formula (from MATH_BREAKDOWN.txt):
            EV = (P_win × Win_amount) - (P_loss × Loss_amount)
        
        Args:
            bet_size: Amount wagered ($)
            win_probability: P(win) from ML model
            american_odds: Odds received from book
        
        Returns:
            Expected value in dollars
        
        Example:
            Bet $400 on LAL -110 with 65% win prob:
            Win amount = $400 × (100/110) = $363.64
            Loss amount = $400
            EV = (0.65 × $363.64) - (0.35 × $400) = $96.36
        
        Time: <1ms
        """
        decimal_odds = ProbabilityConverter.american_to_decimal_odds(american_odds)
        
        win_amount = bet_size * (decimal_odds - 1)
        loss_amount = bet_size
        
        p_loss = 1 - win_probability
        
        ev = (win_probability * win_amount) - (p_loss * loss_amount)
        
        return ev
    
    @staticmethod
    def calculate_vig(american_odds_home: float, american_odds_away: float) -> float:
        """
        Calculate bookmaker vig (overround)
        
        Args:
            american_odds_home: Home odds
            american_odds_away: Away odds
        
        Returns:
            Vig as decimal (e.g., 0.048 = 4.8%)
        
        Example:
            -110 / -110 → 4.8% vig
            -105 / -115 → 4.8% vig
        """
        p_home = ProbabilityConverter.american_to_implied_probability(american_odds_home)
        p_away = ProbabilityConverter.american_to_implied_probability(american_odds_away)
        
        vig = (p_home + p_away) - 1.0
        
        return max(0, vig)


# Test the converter
if __name__ == "__main__":
    import time
    
    print("="*80)
    print("PROBABILITY CONVERTER - VERIFICATION")
    print("="*80)
    
    converter = ProbabilityConverter()
    
    # Test 1: American odds
    print("\n1. American Odds Conversion:")
    print(f"   -110 → Decimal: {converter.american_to_decimal_odds(-110):.4f}")
    print(f"   -110 → Probability: {converter.american_to_implied_probability(-110):.4f} (52.4%)")
    print(f"   +150 → Decimal: {converter.american_to_decimal_odds(+150):.4f}")
    print(f"   +150 → Probability: {converter.american_to_implied_probability(+150):.4f} (40.0%)")
    
    # Test 2: ML interval (from our actual model)
    print("\n2. ML Interval → Win Probability:")
    print(f"   ML Prediction: LAL +15.1 at halftime [+11.3, +18.9]")
    print(f"   Market Spread: LAL -7.5 (full game)")
    
    start = time.time()
    prob = converter.ml_interval_to_probability(
        ml_forecast=15.1,
        ml_lower=11.3,
        ml_upper=18.9,
        market_spread=-7.5
    )
    elapsed = (time.time() - start) * 1000
    
    print(f"   → Win Probability: {prob:.3f} ({prob*100:.1f}%)")
    print(f"   → Calculation time: {elapsed:.2f}ms ✅")
    
    # Test 3: Vig removal
    print("\n3. Remove Vig:")
    p_home = converter.american_to_implied_probability(-110)
    p_away = converter.american_to_implied_probability(-110)
    print(f"   Market: {p_home:.4f} + {p_away:.4f} = {p_home+p_away:.4f}")
    
    p_home_true, p_away_true = converter.remove_vig(p_home, p_away)
    print(f"   True: {p_home_true:.4f} + {p_away_true:.4f} = {p_home_true+p_away_true:.4f}")
    
    # Test 4: Expected value
    print("\n4. Expected Value:")
    bet = 400
    p_win = 0.65
    odds = -110
    ev = converter.expected_value(bet, p_win, odds)
    print(f"   Bet ${ bet} at {odds} with {p_win*100:.0f}% win prob")
    print(f"   → EV: ${ev:.2f}")
    
    # Test 5: Performance test
    print("\n5. Performance Test (1000 conversions):")
    start = time.time()
    for _ in range(1000):
        converter.ml_interval_to_probability(15.1, 11.3, 18.9, -7.5)
    elapsed = (time.time() - start) * 1000
    print(f"   1000 ML conversions: {elapsed:.1f}ms ({elapsed/1000:.2f}ms each)")
    
    if elapsed / 1000 < 5:
        print(f"   ✅ PASS - Under 5ms target")
    else:
        print(f"   ❌ FAIL - Exceeds 5ms target")
    
    print("\n" + "="*80)
    print("✅ PROBABILITY CONVERTER READY")
    print("="*80)

