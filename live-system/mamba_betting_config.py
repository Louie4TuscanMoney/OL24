"""
MAMBA MENTALITY BETTING CONFIGURATION

Optimized for High-Confidence Betting Strategy
Based on Model Breakdown Analysis

Author: Ontologic XYZ
Date: October 20, 2025
"""

# CORE BETTING STRATEGY
BETTING_STRATEGY = "BALANCED"  # Options: HIGH_CONFIDENCE, BALANCED, CONSERVATIVE, ULTRA_SELECTIVE

# STRATEGY PARAMETERS
BETTING_STRATEGIES = {
    'HIGH_CONFIDENCE': {
        'edge_threshold': 5.0,  # ≤5 pts edge
        'expected_games': 439,  # 31.7% of games
        'direction_accuracy': 0.825,  # 82.5%
        'avg_error': 2.57,
        'expected_profit': '+$2,000-3,000',
        'action': 'BET_AGGRESSIVELY'
    },
    'BALANCED': {
        'edge_threshold': 5.0,  # ≥5 pts edge
        'expected_games': 300,  # ~24% of games
        'direction_accuracy': 0.60,  # 60%
        'avg_error': 10.92,
        'expected_profit': '+$1,200-1,600',
        'action': 'BET_STANDARD',
        'kelly_multiplier': 0.25  # Quarter Kelly for safety
    },
    'CONSERVATIVE': {
        'edge_threshold': 7.0,  # ≥7 pts edge
        'expected_games': 140,  # ~10% of games
        'direction_accuracy': 0.694,  # 69.4%
        'avg_error': 11.77,
        'expected_profit': '+$600-900',
        'action': 'BET_CONSERVATIVE',
        'kelly_multiplier': 0.20
    },
    'ULTRA_SELECTIVE': {
        'edge_threshold': 10.0,  # ≥10 pts edge
        'expected_games': 30,  # ~2% of games
        'direction_accuracy': 0.774,  # 77.4%
        'avg_error': 13.31,
        'expected_profit': '+$150-250',
        'action': 'BET_VERY_SELECTIVE',
        'kelly_multiplier': 0.30
    }
}

# GAME TYPE FILTERS
ALLOWED_GAME_TYPES = ['LEAD_HELD']  # Only bet on "Lead Held" games

GAME_TYPE_PERFORMANCE = {
    'LEAD_HELD': {
        'mae': 9.26,
        'pct_of_games': 0.532,  # 53.2%
        'direction_accuracy': 0.72,  # 72%
        'priority': 'HIGH'
    },
    'CLOSE': {
        'mae': 9.85,
        'pct_of_games': 0.26,
        'direction_accuracy': 0.65,
        'priority': 'LOW'  # Don't bet
    },
    'COMEBACK': {
        'mae': 13.7,
        'pct_of_games': 0.15,
        'direction_accuracy': 0.55,
        'priority': 'NEVER'  # Never bet
    }
}

# KEY FEATURES (MOST IMPORTANT)
KEY_FEATURES = {
    'pattern_min': 5.25,  # Most important feature
    'pattern_max': 4.73,  # Second most important
}

# PERFORMANCE THRESHOLDS
CONFIDENCE_ZONES = {
    'HIGH': {
        'threshold': 5.0,  # ≤5 pts error
        'pct_of_games': 0.317,  # 31.7%
        'avg_error': 2.57,
        'direction_accuracy': 0.825,
        'action': 'BET'
    },
    'MEDIUM': {
        'threshold': 12.0,  # 5-12 pts error
        'pct_of_games': 0.32,  # ~32%
        'avg_error': 8.5,
        'direction_accuracy': 0.65,
        'action': 'CONSIDER'
    },
    'LOW': {
        'threshold': 999,  # >12 pts error
        'pct_of_games': 0.36,
        'avg_error': 18.0,
        'direction_accuracy': 0.55,
        'action': 'SKIP'
    }
}

# BIAS ADJUSTMENTS
BIAS_BY_GAME_STATE = {
    'VERY_CLOSE': {  # ≤3 pts
        'bias': +0.20,
        'adjustment': 'Slight over-prediction'
    },
    'CLOSE': {  # 4-7 pts
        'bias': +0.60,
        'adjustment': 'Moderate over-prediction'
    }
}

# BRANCH PERFORMANCE
DUAL_BRANCH = {
    'HALFTIME': {
        'mae': 5.4,
        'edge_over_baseline': 0.40,  # 40%
        'betting_window': 'Q2 6:00',
        'opportunities_per_game': 1
    },
    'FINAL': {
        'mae': 9.9,
        'edge_over_baseline': 0.22,  # 22%
        'betting_window': 'Q2 6:00',
        'opportunities_per_game': 1
    },
    'TOTAL_OPPORTUNITIES_PER_GAME': 2
}

# RISK MANAGEMENT
RISK_PARAMETERS = {
    'bankroll': 1000,  # $1,000 starting
    'max_positions': 5,
    'daily_loss_limit_pct': 0.10,  # 10% = $100
    'weekly_loss_limit_pct': 0.20,  # 20% = $200
    'max_bet_pct': 0.05,  # 5% = $50 max per bet
    'kelly_fraction': 0.25,  # Quarter Kelly (conservative)
    'min_edge': 5.0,  # Minimum 5 pts edge to bet
    'min_confidence': 0.60  # Minimum 60% P(Win)
}

# PERFORMANCE QUARTILES
PERFORMANCE_QUARTILES = {
    '25th': 4.04,  # Top quarter ⭐⭐⭐
    '50th': 8.59,  # Median (typical)
    '75th': 12.0,  # Below average
}

# EXPECTED OUTCOMES (BALANCED STRATEGY)
EXPECTED_OUTCOMES = {
    'games_per_season': 1230,  # 82 games × 15 teams / 2
    'bet_opportunities': 300,  # ~24% of games
    'win_rate': 0.60,  # 60%
    'avg_profit_per_bet': 4.80,  # $4.80 per bet
    'season_profit': 71000,  # $71,000 (optimistic)
    'realistic_profit': 1200,  # $1,200-1,600 (conservative with $1k bankroll)
}


def get_current_strategy():
    """Get currently configured betting strategy"""
    return BETTING_STRATEGIES[BETTING_STRATEGY]


def should_bet_on_game(edge: float, game_type: str, confidence: float) -> tuple[bool, str]:
    """
    Determine if we should bet on a game
    
    Args:
        edge: Predicted edge in points
        game_type: Type of game (LEAD_HELD, CLOSE, COMEBACK)
        confidence: Win probability
        
    Returns:
        (should_bet, reason)
    """
    strategy = get_current_strategy()
    
    # Check game type filter
    if game_type not in ALLOWED_GAME_TYPES:
        return False, f"Game type {game_type} not in allowed list (only {ALLOWED_GAME_TYPES})"
    
    # Check edge threshold
    if edge < strategy['edge_threshold']:
        return False, f"Edge {edge:.1f} below threshold {strategy['edge_threshold']}"
    
    # Check confidence
    if confidence < RISK_PARAMETERS['min_confidence']:
        return False, f"Confidence {confidence:.1%} below minimum {RISK_PARAMETERS['min_confidence']:.1%}"
    
    # All checks passed
    return True, f"✅ {strategy['action']}: Edge {edge:.1f} pts, Confidence {confidence:.1%}"


def calculate_bet_size(edge: float, confidence: float, bankroll: float) -> float:
    """
    Calculate optimal bet size using Kelly criterion
    
    Args:
        edge: Predicted edge in points
        confidence: Win probability
        bankroll: Current bankroll
        
    Returns:
        Bet size in dollars
    """
    strategy = get_current_strategy()
    kelly_multiplier = strategy.get('kelly_multiplier', 0.25)
    
    # Kelly formula: f = (bp - q) / b
    # Where: b = odds (assume -110 = 0.909), p = confidence, q = 1-p
    odds_decimal = 0.909  # -110 American odds
    kelly_fraction = (odds_decimal * confidence - (1 - confidence)) / odds_decimal
    
    # Apply fractional Kelly
    kelly_fraction = kelly_fraction * kelly_multiplier
    
    # Calculate bet size
    bet_size = bankroll * kelly_fraction
    
    # Apply maximum bet constraint
    max_bet = bankroll * RISK_PARAMETERS['max_bet_pct']
    bet_size = min(bet_size, max_bet)
    
    # Minimum bet
    bet_size = max(bet_size, 10)  # Minimum $10
    
    return round(bet_size, 2)


# TESTING
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🏀 MAMBA MENTALITY BETTING CONFIGURATION")
    print("="*80 + "\n")
    
    print(f"Current Strategy: {BETTING_STRATEGY}")
    strategy = get_current_strategy()
    print(f"  • Edge Threshold: ≥{strategy['edge_threshold']} pts")
    print(f"  • Expected Games: {strategy['expected_games']}")
    print(f"  • Direction Accuracy: {strategy['direction_accuracy']:.1%}")
    print(f"  • Expected Profit: {strategy['expected_profit']}")
    print()
    
    print("Game Type Filter:")
    print(f"  • Allowed: {ALLOWED_GAME_TYPES}")
    print(f"  • Lead Held: {GAME_TYPE_PERFORMANCE['LEAD_HELD']['mae']:.2f} MAE, {GAME_TYPE_PERFORMANCE['LEAD_HELD']['direction_accuracy']:.1%} accuracy")
    print()
    
    print("Risk Management:")
    print(f"  • Bankroll: ${RISK_PARAMETERS['bankroll']:,}")
    print(f"  • Max Bet: {RISK_PARAMETERS['max_bet_pct']:.1%} = ${RISK_PARAMETERS['bankroll'] * RISK_PARAMETERS['max_bet_pct']:.0f}")
    print(f"  • Daily Loss Limit: {RISK_PARAMETERS['daily_loss_limit_pct']:.1%} = ${RISK_PARAMETERS['bankroll'] * RISK_PARAMETERS['daily_loss_limit_pct']:.0f}")
    print(f"  • Kelly Fraction: {strategy.get('kelly_multiplier', 0.25):.2f} (conservative)")
    print()
    
    # Test examples
    print("Example Bet Calculations:")
    print()
    
    test_cases = [
        (6.0, 0.703, "LEAD_HELD"),  # High confidence
        (5.5, 0.65, "LEAD_HELD"),   # Balanced
        (8.0, 0.72, "LEAD_HELD"),   # Conservative
        (4.0, 0.70, "LEAD_HELD"),   # Below threshold
        (7.0, 0.68, "CLOSE"),       # Wrong game type
    ]
    
    for edge, conf, game_type in test_cases:
        should_bet, reason = should_bet_on_game(edge, game_type, conf)
        
        if should_bet:
            bet_size = calculate_bet_size(edge, conf, RISK_PARAMETERS['bankroll'])
            print(f"✅ Edge: {edge:.1f}, P(Win): {conf:.1%}, Type: {game_type}")
            print(f"   Bet Size: ${bet_size:.2f}")
            print(f"   Reason: {reason}")
        else:
            print(f"❌ Edge: {edge:.1f}, P(Win): {conf:.1%}, Type: {game_type}")
            print(f"   Reason: {reason}")
        print()
    
    print("="*80)
    print("✅ MAMBA MENTALITY OPTIMIZED FOR BALANCED BETTING")
    print("="*80)

