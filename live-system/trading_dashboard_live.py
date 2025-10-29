"""
TRADING DASHBOARD - Interactive Sports Betting Interface

Features:
- Live Mamba predictions
- Interactive odds input
- Bankroll management
- EV calculation
- Bet sizing (Kelly Criterion)
- Real-time P&L tracking
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import psycopg2
import os
from datetime import datetime

router = APIRouter()
DATABASE_URL = os.getenv('DATABASE_URL')


class BetInput(BaseModel):
    """User bet input"""
    game_id: str
    bet_type: str  # 'spread', 'total', 'moneyline', '2h_spread'
    side: str  # 'home', 'away', 'over', 'under'
    odds: float  # American odds (e.g., -110, +150)
    stake: float  # Amount to bet
    book: Optional[str] = 'DraftKings'


class BetAnalysis(BaseModel):
    """Bet analysis with EV"""
    bet_input: BetInput
    mamba_prediction: Optional[float]
    mamba_confidence: Optional[float]
    implied_probability: float
    mamba_probability: Optional[float]
    expected_value: float
    expected_profit: float
    kelly_stake: Optional[float]
    recommendation: str
    risk_level: str


@router.get("/api/trading/live-opportunities")
async def get_live_betting_opportunities():
    """
    Get ALL live games with Mamba predictions and betting opportunities
    
    Returns:
        List of games with predictions, odds, and EV calculations
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Get all games with Mamba predictions
        cur.execute("""
            SELECT 
                mgc.game_id,
                mgc.prediction,
                mgc.confidence,
                mgc.home_team_id,
                mgc.away_team_id,
                mgc.home_score,
                mgc.away_score,
                mgc.current_margin,
                mgc.triggered_at,
                t1.abbreviation as home_abbr,
                t2.abbreviation as away_abbr
            FROM mamba_game_cache mgc
            LEFT JOIN teams t1 ON mgc.home_team_id = t1.team_id
            LEFT JOIN teams t2 ON mgc.away_team_id = t2.team_id
            WHERE mgc.final_home_score IS NULL  -- Only live/active games
            AND mgc.triggered_at > NOW() - INTERVAL '4 hours'
            ORDER BY mgc.triggered_at DESC
        """)
        
        opportunities = []
        
        for row in cur.fetchall():
            game_id = row[0]
            prediction = float(row[1])
            confidence = float(row[2])
            
            # Calculate expected probabilities
            home_win_prob = calculate_win_probability(prediction, confidence)
            away_win_prob = 1 - home_win_prob
            
            # Example odds (in production, fetch from odds API)
            home_spread_odds = -110
            away_spread_odds = -110
            
            # Calculate EV for each bet type
            home_spread_ev = calculate_ev(
                prediction, 
                confidence, 
                home_spread_odds, 
                'home'
            )
            
            away_spread_ev = calculate_ev(
                prediction * -1,  # Flip for away
                confidence,
                away_spread_odds,
                'away'
            )
            
            opportunities.append({
                'game_id': game_id,
                'home_team': row[9],
                'away_team': row[10],
                'current_score': f"{row[5]}-{row[6]}",
                'current_margin': row[7],
                'mamba_prediction': prediction,
                'mamba_confidence': confidence,
                'home_win_probability': round(home_win_prob * 100, 1),
                'away_win_probability': round(away_win_prob * 100, 1),
                'opportunities': [
                    {
                        'type': 'home_spread',
                        'odds': home_spread_odds,
                        'ev': round(home_spread_ev, 2),
                        'recommendation': 'BET' if home_spread_ev > 2 else 'PASS'
                    },
                    {
                        'type': 'away_spread',
                        'odds': away_spread_odds,
                        'ev': round(away_spread_ev, 2),
                        'recommendation': 'BET' if away_spread_ev > 2 else 'PASS'
                    }
                ],
                'triggered_at': row[8].isoformat()
            })
        
        cur.close()
        conn.close()
        
        return {
            'count': len(opportunities),
            'opportunities': opportunities,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/trading/analyze-bet")
async def analyze_bet(bet: BetInput) -> BetAnalysis:
    """
    Analyze a specific bet with Mamba prediction
    
    Calculate:
    - Expected Value (EV)
    - Kelly Criterion stake
    - Win probability
    - Risk assessment
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Get Mamba prediction for this game
        cur.execute("""
            SELECT prediction, confidence
            FROM mamba_game_cache
            WHERE game_id = %s
        """, (bet.game_id,))
        
        result = cur.fetchone()
        
        if not result:
            raise HTTPException(404, "No Mamba prediction for this game")
        
        mamba_prediction = float(result[0])
        mamba_confidence = float(result[1])
        
        cur.close()
        conn.close()
        
        # Calculate implied probability from odds
        implied_prob = odds_to_probability(bet.odds)
        
        # Calculate Mamba's win probability
        mamba_prob = calculate_bet_win_probability(
            mamba_prediction,
            mamba_confidence,
            bet.bet_type,
            bet.side
        )
        
        # Calculate EV
        ev = calculate_expected_value(
            mamba_prob,
            implied_prob,
            bet.odds,
            bet.stake
        )
        
        expected_profit = ev - bet.stake
        
        # Calculate Kelly Criterion
        kelly = calculate_kelly(mamba_prob, bet.odds)
        
        # Risk assessment
        risk_level = assess_risk(ev, mamba_confidence, kelly)
        
        # Recommendation
        recommendation = generate_recommendation(ev, mamba_confidence, kelly)
        
        return BetAnalysis(
            bet_input=bet,
            mamba_prediction=mamba_prediction,
            mamba_confidence=mamba_confidence,
            implied_probability=round(implied_prob * 100, 2),
            mamba_probability=round(mamba_prob * 100, 2),
            expected_value=round(ev, 2),
            expected_profit=round(expected_profit, 2),
            kelly_stake=round(kelly * 1000, 2) if kelly > 0 else 0,  # Assume $1000 bankroll
            recommendation=recommendation,
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/trading/place-bet")
async def track_bet(bet: BetInput):
    """
    Track a placed bet in the database
    
    Stores bet details for P&L tracking
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Get Mamba prediction
        cur.execute("""
            SELECT prediction, confidence
            FROM mamba_game_cache
            WHERE game_id = %s
        """, (bet.game_id,))
        
        result = cur.fetchone()
        mamba_prediction = float(result[0]) if result else None
        mamba_confidence = float(result[1]) if result else None
        
        # Calculate EV
        implied_prob = odds_to_probability(bet.odds)
        mamba_prob = calculate_bet_win_probability(
            mamba_prediction,
            mamba_confidence,
            bet.bet_type,
            bet.side
        )
        ev = calculate_expected_value(mamba_prob, implied_prob, bet.odds, bet.stake)
        
        # Store bet
        cur.execute("""
            INSERT INTO tracked_bets (
                game_id, bet_type, side, odds, stake, book,
                mamba_prediction, mamba_confidence,
                expected_value, placed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            RETURNING bet_id
        """, (
            bet.game_id, bet.bet_type, bet.side, bet.odds, bet.stake, bet.book,
            mamba_prediction, mamba_confidence, ev
        ))
        
        bet_id = cur.fetchone()[0]
        
        conn.commit()
        cur.close()
        conn.close()
        
        return {
            'bet_id': bet_id,
            'status': 'tracked',
            'expected_value': round(ev, 2),
            'message': f'Bet tracked successfully. EV: ${ev:.2f}'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/trading/performance")
async def get_trading_performance():
    """
    Get overall trading/betting performance
    
    Returns:
    - Total bets placed
    - Win rate
    - Total P&L
    - ROI
    - Avg EV
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Get overall stats
        cur.execute("""
            SELECT 
                COUNT(*) as total_bets,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(profit_loss) as total_pl,
                SUM(stake) as total_staked,
                AVG(expected_value) as avg_ev
            FROM tracked_bets
            WHERE result IS NOT NULL
        """)
        
        stats = cur.fetchone()
        
        # Get recent bets
        cur.execute("""
            SELECT 
                game_id, bet_type, side, odds, stake,
                expected_value, result, profit_loss, placed_at
            FROM tracked_bets
            ORDER BY placed_at DESC
            LIMIT 20
        """)
        
        recent_bets = []
        for row in cur.fetchall():
            recent_bets.append({
                'game_id': row[0],
                'bet_type': row[1],
                'side': row[2],
                'odds': row[3],
                'stake': float(row[4]),
                'ev': float(row[5]),
                'result': row[6],
                'profit_loss': float(row[7]) if row[7] else None,
                'placed_at': row[8].isoformat()
            })
        
        cur.close()
        conn.close()
        
        total_bets = stats[0] or 0
        wins = stats[1] or 0
        total_pl = float(stats[2]) if stats[2] else 0
        total_staked = float(stats[3]) if stats[3] else 1
        avg_ev = float(stats[4]) if stats[4] else 0
        
        win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
        roi = (total_pl / total_staked * 100) if total_staked > 0 else 0
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': total_bets - wins,
            'win_rate': round(win_rate, 1),
            'total_profit_loss': round(total_pl, 2),
            'total_staked': round(total_staked, 2),
            'roi': round(roi, 1),
            'avg_expected_value': round(avg_ev, 2),
            'recent_bets': recent_bets
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def odds_to_probability(american_odds: float) -> float:
    """Convert American odds to implied probability"""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def calculate_win_probability(prediction: float, confidence: float) -> float:
    """Calculate win probability from Mamba prediction"""
    # Simple logistic transformation
    # prediction is spread, confidence is 0-100
    prob = 1 / (1 + 2.718 ** (-prediction / 10))
    
    # Adjust by confidence
    confidence_factor = confidence / 100
    prob = prob * confidence_factor + 0.5 * (1 - confidence_factor)
    
    return prob


def calculate_bet_win_probability(
    prediction: float,
    confidence: float,
    bet_type: str,
    side: str
) -> float:
    """Calculate probability of specific bet winning"""
    base_prob = calculate_win_probability(prediction, confidence)
    
    if bet_type == 'spread':
        if side == 'home':
            return base_prob
        else:
            return 1 - base_prob
    elif bet_type == '2h_spread':
        # For 2H, use higher uncertainty
        return base_prob * 0.9 + 0.5 * 0.1
    else:
        return 0.5  # Default


def calculate_expected_value(
    win_prob: float,
    implied_prob: float,
    odds: float,
    stake: float
) -> float:
    """Calculate expected value of bet"""
    # Calculate payout
    if odds > 0:
        payout = stake * (odds / 100)
    else:
        payout = stake * (100 / abs(odds))
    
    # EV = (win_prob * payout) - ((1 - win_prob) * stake)
    ev = (win_prob * (payout + stake)) - ((1 - win_prob) * stake)
    
    return ev


def calculate_ev(prediction: float, confidence: float, odds: float, side: str) -> float:
    """Calculate EV for a bet"""
    win_prob = calculate_win_probability(prediction if side == 'home' else -prediction, confidence)
    implied_prob = odds_to_probability(odds)
    
    # Edge = win_prob - implied_prob
    edge = win_prob - implied_prob
    
    # EV as percentage
    ev_pct = edge * 100
    
    return ev_pct


def calculate_kelly(win_prob: float, odds: float) -> float:
    """Calculate Kelly Criterion bet size"""
    if odds > 0:
        decimal_odds = 1 + (odds / 100)
    else:
        decimal_odds = 1 + (100 / abs(odds))
    
    kelly = (win_prob * decimal_odds - 1) / (decimal_odds - 1)
    
    # Use fractional Kelly (1/4 Kelly for safety)
    return max(0, kelly * 0.25)


def assess_risk(ev: float, confidence: float, kelly: float) -> str:
    """Assess risk level of bet"""
    if ev < 0:
        return 'HIGH'
    elif ev > 5 and confidence > 70 and kelly > 0.02:
        return 'LOW'
    elif ev > 2 and confidence > 60:
        return 'MEDIUM'
    else:
        return 'HIGH'


def generate_recommendation(ev: float, confidence: float, kelly: float) -> str:
    """Generate bet recommendation"""
    if ev < 0:
        return 'AVOID - Negative EV'
    elif ev > 5 and confidence > 75:
        return 'STRONG BET - High EV & Confidence'
    elif ev > 3 and confidence > 65:
        return 'BET - Positive EV'
    elif ev > 1 and confidence > 60:
        return 'SMALL BET - Marginal EV'
    else:
        return 'PASS - Low EV or Confidence'

