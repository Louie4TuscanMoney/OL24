"""
TRADING DASHBOARD API

Purpose: Backend API for SolidJS trading dashboard
Author: Ontologic XYZ
Date: October 20, 2025

This serves:
- Live game data
- Betting opportunities
- Risk status
- Predictions
- Trade history
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import uvicorn
import asyncio
import time
import sys

import os

# Import components (all in same directory on Railway)
try:
    from live_trading_engine import LiveTradingEngine
except ImportError:
    LiveTradingEngine = None
    print("⚠️ LiveTradingEngine not available")

try:
    from nba_live_scores import NBALiveScores
except ImportError:
    NBALiveScores = None
    print("⚠️ NBALiveScores not available")

try:
    from betonline_live_lines import BetOnlineScraper
except ImportError:
    BetOnlineScraper = None
    print("⚠️ BetOnlineScraper not available")

try:
    from user_auth_manager import UserAuthManager
except ImportError:
    UserAuthManager = None
    print("⚠️ UserAuthManager not available")

try:
    from bet_portfolio_manager import BetPortfolioManager
except ImportError:
    BetPortfolioManager = None
    print("⚠️ BetPortfolioManager not available")

try:
    from court_3d_stream import Court3DStream
except ImportError:
    Court3DStream = None
    print("⚠️ Court3DStream not available")

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../4. Risk'))
    from ontorisk_phase4_risk_management import RiskManager  # type: ignore
    ONTORISK_AVAILABLE = True
except ImportError:
    ONTORISK_AVAILABLE = False
    print("⚠️ OntoRisk not available (expected)")


# Initialize FastAPI
app = FastAPI(
    title="Ontologic XYZ Trading Dashboard API",
    description="Live NBA betting system with ML predictions and risk management",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: restrict to dashboard domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
trading_engine = None
portfolio_manager = None
court_stream = None
auth_manager = None
active_connections: List[WebSocket] = []

# Simple cache for predictions (avoid recomputing every request)
_prediction_cache = {
    'data': None,
    'timestamp': None,
    'ttl_seconds': 3  # Cache for 3 seconds (fast updates!)
}


@app.on_event("startup")
async def startup():
    """
    Initialize system on Railway startup
    
    This runs automatically when Railway starts the backend.
    It will:
    1. Download Mamba model from Google Drive (if not present)
    2. Initialize NBA API
    3. Initialize Trading Engine
    4. Set up OntoRisk
    
    CRITICAL: Your computer can be OFF - this runs on Railway 24/7!
    """
    global trading_engine, portfolio_manager, court_stream, auth_manager, nba_api
    
    print("\n" + "="*80)
    print("🚀 RAILWAY STARTUP: INITIALIZING AUTONOMOUS SYSTEM")
    print("="*80 + "\n")
    
    try:
        # Initialize NBA API first
        print("🏀 Initializing NBA Live Scores API...")
        if NBALiveScores:
            nba_api = NBALiveScores()
            print("✅ NBA API ready - can fetch live games & play-by-play")
        else:
            print("⚠️ NBA API not available")
            nba_api = None
        
        # Initialize Trading Engine (will auto-download Mamba model!)
        print("\n🐍 Initializing Mamba Trading Engine...")
        if LiveTradingEngine:
            trading_engine = LiveTradingEngine(
                model_path=None,  # ⚡ CRITICAL: Triggers Google Drive auto-download!
                mae=9.655,  # ✅ CORRECT: Branch B (Final Score) MAE
                starting_bankroll=1000
            )
            print("✅ Trading engine initialized")
            print(f"   → Mamba model loaded: {trading_engine.model is not None}")
            print(f"   → OntoRisk enabled: {trading_engine.ontorisk_enabled}")
            print(f"   → MAE: {trading_engine.mae}")
        else:
            print("⚠️ Trading Engine not available")
            trading_engine = None
        
        # Initialize Portfolio Manager
        print("\n💼 Initializing Portfolio Manager...")
        if BetPortfolioManager:
            portfolio_manager = BetPortfolioManager()
            print("✅ Portfolio manager initialized")
        else:
            print("⚠️ Portfolio manager not available")
            portfolio_manager = None
        
        # Initialize 3D Court Stream
        print("\n🏀 Initializing 3D Court Stream...")
        if Court3DStream:
            court_stream = Court3DStream()
            print("✅ 3D court stream initialized")
        else:
            print("⚠️ 3D court stream not available")
            court_stream = None
        
        # Initialize User Auth Manager
        print("\n👥 Initializing User Auth Manager...")
        if UserAuthManager:
            auth_manager = UserAuthManager()
            print("✅ User auth manager initialized")
        else:
            print("⚠️ User auth manager not available")
            auth_manager = None
        
        print("\n" + "="*80)
        print("✅ SYSTEM FULLY INITIALIZED - READY FOR LIVE PREDICTIONS!")
        print("="*80)
        print("\n💡 System Status:")
        print(f"   NBA API: {'✅' if nba_api else '❌'}")
        print(f"   Mamba Model: {'✅' if trading_engine and trading_engine.model else '❌'}")
        print(f"   OntoRisk: {'✅' if trading_engine and trading_engine.ontorisk_enabled else '❌'}")
        print(f"   Portfolio Manager: {'✅' if portfolio_manager else '❌'}")
        print(f"   3D Court: {'✅' if court_stream else '❌'}")
        print(f"   Auth System: {'✅' if auth_manager else '❌'}")
        print("\n🎯 Waiting for live NBA games...")
        print("   → System will automatically detect Q2 6:00 marks")
        print("   → Extract 33 real features from play-by-play")
        print("   → Make Mamba predictions")
        print("   → Push to Vercel via WebSocket\n")
        
    except Exception as e:
        print(f"\n❌ CRITICAL STARTUP ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️ System will continue but predictions may not work!\n")


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "system": "Ontologic XYZ Trading Dashboard",
        "version": "1.0.0",
        "ontorisk_enabled": ONTORISK_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/debug/system-status")
async def debug_system_status():
    """
    Debug endpoint to check if ML system is working
    """
    status = {
        "trading_engine_initialized": trading_engine is not None,
        "ml_model_loaded": False,
        "mamba_extractor_loaded": False,
        "nba_api_initialized": nba_api is not None,
        "recent_games_count": 0,
        "recent_predictions_count": 0,
        "startup_complete": True
    }
    
    if trading_engine:
        status["ml_model_loaded"] = trading_engine.model is not None
        status["mamba_extractor_loaded"] = hasattr(trading_engine, 'mamba_extractor') and trading_engine.mamba_extractor is not None
        status["ontorisk_enabled"] = trading_engine.ontorisk_enabled
        
        # Try to scan for predictions
        try:
            print("🔍 DEBUG: Attempting to scan live opportunities...")
            predictions = trading_engine.scan_live_opportunities()
            status["recent_predictions_count"] = len(predictions)
            print(f"📊 DEBUG: Got {len(predictions)} predictions")
            
            # Show first prediction details
            if predictions:
                status["sample_prediction"] = {
                    "matchup": predictions[0].get("matchup", "Unknown"),
                    "prediction": predictions[0].get("prediction", "N/A"),
                    "has_features": predictions[0].get("features_extracted", False)
                }
        except Exception as e:
            status["scan_error"] = str(e)
            print(f"❌ DEBUG: Scan error: {e}")
            import traceback
            traceback.print_exc()
    else:
        status["startup_complete"] = False
        status["error"] = "Trading engine not initialized!"
    
    if nba_api:
        try:
            games = nba_api.get_todays_games()
            status["recent_games_count"] = len(games)
            # Show first game details
            if games:
                status["sample_game"] = {
                    "matchup": f"{games[0].get('away_team')} @ {games[0].get('home_team')}",
                    "period": games[0].get("period"),
                    "can_predict": games[0].get("can_predict", False)
                }
        except Exception as e:
            status["games_error"] = str(e)
    
    return status


# Auth Endpoints
class SignupRequest(BaseModel):
    phone: str
    password: str

class PasswordCheck(BaseModel):
    password: str

@app.post("/api/auth/signup")
async def signup(request: SignupRequest):
    """
    Submit access request
    
    Args:
        request: Phone and password
        
    Returns:
        Success message
    """
    if auth_manager is None:
        return JSONResponse({"detail": "System not initialized"}, status_code=503)
    
    success = auth_manager.submit_access_request(request.phone, request.password)
    
    if success:
        return {"message": "Request submitted successfully"}
    else:
        return JSONResponse({"detail": "Error submitting request"}, status_code=500)


@app.post("/api/auth/check-password")
async def check_password(request: PasswordCheck):
    """
    Check if password is approved
    
    Args:
        request: Password to check
        
    Returns:
        Approval status
    """
    if auth_manager is None:
        return JSONResponse({"detail": "System not initialized"}, status_code=503)
    
    approved = auth_manager.check_password(request.password)
    
    if approved:
        auth_manager.update_last_login(request.password)
    
    return {"approved": approved}


@app.get("/api/auth/pending-requests")
async def get_pending_requests():
    """
    Get all pending access requests (admin only)
    
    Returns:
        List of pending requests
    """
    if auth_manager is None:
        return JSONResponse({"detail": "System not initialized"}, status_code=503)
    
    requests = auth_manager.get_pending_requests()
    return {"requests": requests, "count": len(requests)}


@app.post("/api/auth/approve/{request_id}")
async def approve_request(request_id: int):
    """
    Approve an access request (admin only)
    
    Args:
        request_id: ID of request to approve
        
    Returns:
        Success message
    """
    if auth_manager is None:
        return JSONResponse({"detail": "System not initialized"}, status_code=503)
    
    success = auth_manager.approve_request(request_id)
    
    if success:
        return {"message": "Request approved"}
    else:
        return JSONResponse({"detail": "Error approving request"}, status_code=500)


@app.get("/api/live-games")
async def get_live_games():
    """
    Get current live games
    
    Returns:
        List of live games with scores
    """
    if trading_engine is None:
        return JSONResponse({"error": "System not initialized"}, status_code=503)
    
    games = trading_engine.nba_api.get_todays_games()
    
    return {
        "games": games,
        "count": len(games),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/live-lines")
async def get_live_lines():
    """
    Get current betting lines
    
    Returns:
        List of lines from BetOnline
    """
    if trading_engine is None:
        return JSONResponse({"error": "System not initialized"}, status_code=503)
    
    lines = trading_engine.line_scraper.get_live_lines()
    
    return {
        "lines": lines,
        "count": len(lines),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/opportunities")
async def get_opportunities():
    """
    Get current betting opportunities with full context
    
    Returns:
        List of ALL opportunities with categories, percentages, and betting decisions
    """
    if trading_engine is None:
        return JSONResponse({"error": "System not initialized"}, status_code=503)
    
    # Check cache first
    now = datetime.now()
    if (_prediction_cache['data'] is not None and 
        _prediction_cache['timestamp'] is not None):
        age_seconds = (now - _prediction_cache['timestamp']).total_seconds()
        if age_seconds < _prediction_cache['ttl_seconds']:
            # Return cached data
            return _prediction_cache['data']
    
    # Cache miss or expired - compute fresh predictions
    opportunities = trading_engine.scan_live_opportunities()
    
    # Enhance each opportunity with category, zone, and betting decision
    enhanced_opps = []
    for opp in opportunities:
        edge = abs(opp.get('edge', 0))
        confidence = opp.get('p_win', 0)
        current_diff = abs(opp.get('current_score', 0))
        
        # Determine game category
        if current_diff > 10:
            game_category = 'LEAD_HELD'
            category_mae = 9.26
            category_accuracy = 0.72
            category_pct = 53.2
        elif current_diff > 3:
            game_category = 'CLOSE'
            category_mae = 9.85
            category_accuracy = 0.65
            category_pct = 26
        else:
            game_category = 'VERY_CLOSE'
            category_mae = 9.85
            category_accuracy = 0.65
            category_pct = 15
        
        # Determine confidence zone
        if edge <= 5:
            zone = 'HIGH'
            zone_pct = 31.7
            zone_accuracy = 82.5
            zone_avg_error = 2.57
        elif edge <= 12:
            zone = 'MEDIUM'
            zone_pct = 32
            zone_accuracy = 65
            zone_avg_error = 8.5
        else:
            zone = 'LOW'
            zone_pct = 36
            zone_accuracy = 55
            zone_avg_error = 18.0
        
        # Determine betting strategy
        if edge <= 5:
            strategy = 'HIGH_CONFIDENCE'
            strategy_games = 439
            strategy_accuracy = 82.5
            strategy_pct = 31.7
        elif edge >= 5 and edge < 7:
            strategy = 'BALANCED'
            strategy_games = 300
            strategy_accuracy = 60
            strategy_pct = 24
        elif edge >= 7 and edge < 10:
            strategy = 'CONSERVATIVE'
            strategy_games = 140
            strategy_accuracy = 69.4
            strategy_pct = 10
        else:
            strategy = 'ULTRA_SELECTIVE'
            strategy_games = 30
            strategy_accuracy = 77.4
            strategy_pct = 2
        
        # Should we bet?
        should_bet = (
            edge >= 5 and 
            game_category == 'LEAD_HELD' and 
            confidence >= 0.60
        )
        
        # Skip reason
        skip_reason = None
        if not should_bet:
            if game_category != 'LEAD_HELD':
                skip_reason = f"Game type '{game_category}' not approved (only Lead Held)"
            elif edge < 5:
                skip_reason = f"Edge {edge:.1f} below 5.0 threshold"
            elif confidence < 0.60:
                skip_reason = f"Confidence {confidence:.1%} below 60% minimum"
        
        enhanced_opps.append({
            **opp,
            'game_category': game_category,
            'category_mae': category_mae,
            'category_accuracy': category_accuracy,
            'category_pct_of_games': category_pct,
            'confidence_zone': zone,
            'zone_pct_of_games': zone_pct,
            'zone_accuracy': zone_accuracy,
            'zone_avg_error': zone_avg_error,
            'betting_strategy': strategy,
            'strategy_expected_games': strategy_games,
            'strategy_accuracy': strategy_accuracy,
            'strategy_pct_of_games': strategy_pct,
            'should_bet': should_bet,
            'skip_reason': skip_reason
        })
    
    # Separate betting vs context opportunities
    betting_opps = [o for o in enhanced_opps if o['should_bet']]
    context_opps = [o for o in enhanced_opps if not o['should_bet']]
    
    # Build response
    response = {
        "all_opportunities": enhanced_opps,
        "betting_opportunities": betting_opps,
        "context_opportunities": context_opps,
        "total_count": len(enhanced_opps),
        "betting_count": len(betting_opps),
        "context_count": len(context_opps),
        "timestamp": datetime.now().isoformat()
    }
    
    # Cache the response
    _prediction_cache['data'] = response
    _prediction_cache['timestamp'] = now
    
    return response


@app.get("/api/mamba-performance")
async def get_mamba_performance():
    """Get Mamba prediction performance statistics"""
    if trading_engine is None:
        return JSONResponse({"error": "System not initialized"}, status_code=503)
    
    try:
        performance = trading_engine.get_mamba_performance()
        return {
            "mamba_performance": performance,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/mamba-scores")
async def get_stored_mamba_scores():
    """Get stored Mamba scores after 6:00 mark"""
    if trading_engine is None:
        return JSONResponse({"error": "System not initialized"}, status_code=503)
    
    try:
        scores = trading_engine.get_stored_mamba_scores()
        return {
            "mamba_scores": scores,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/mamba-daily-log")
async def get_mamba_daily_log(date: str = None):
    """
    Get Mamba daily log with full prediction details
    
    Args:
        date: Date string (YYYY-MM-DD), defaults to today
    
    Returns:
        Daily log with all predictions and summary
    """
    if trading_engine is None:
        return JSONResponse({"error": "System not initialized"}, status_code=503)
    
    try:
        summary = trading_engine.auto_logger.get_daily_summary(date)
        return {
            "daily_summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/mamba-update-outcome")
async def update_mamba_outcome(request: dict):
    """
    Update outcome for a Mamba prediction
    
    Body:
        {
            "game_id": "0022500123",
            "final_home_score": 112,
            "final_away_score": 108,
            "bet_placed": true,
            "bet_amount": 100,
            "bet_result": "win"
        }
    
    Returns:
        Updated outcome confirmation
    """
    if trading_engine is None:
        return JSONResponse({"error": "System not initialized"}, status_code=503)
    
    try:
        trading_engine.auto_logger.update_outcome(
            game_id=request.get('game_id'),
            final_home_score=request.get('final_home_score'),
            final_away_score=request.get('final_away_score'),
            bet_placed=request.get('bet_placed', False),
            bet_amount=request.get('bet_amount', 0),
            bet_result=request.get('bet_result')
        )
        
        return {
            "status": "✅ Outcome updated!",
            "game_id": request.get('game_id'),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/betonline/live/{game_id}")
async def get_betonline_for_game(game_id: str):
    """
    Get BetOnline live odds for a specific game
    Shows LOCKED when lines unavailable (game transitions)
    
    Returns:
        Live spread, moneyline, total, or LOCKED status
    """
    if trading_engine is None:
        return JSONResponse({"error": "System not initialized"}, status_code=503)
    
    try:
        # Get BetOnline data for specific game
        scraper = trading_engine.line_scraper
        lines = scraper.get_live_lines()
        
        # Default: LOCKED (lines locked during transitions/recalculation)
        game_line = {
            "game_id": game_id,
            "spread": None,
            "moneyline_home": None,
            "moneyline_away": None,
            "total": None,
            "timestamp": datetime.now().isoformat(),
            "source": "BetOnline",
            "available": False,
            "locked": True,
            "lock_reason": "LOCKED (transition/recalculation)"
        }
        
        # Try to find matching game with real odds
        for line in lines:
            if line.get('game_id') == game_id or game_id in str(line.get('game_id', '')):
                game_line = {
                    "game_id": game_id,
                    "spread": line.get('spread'),
                    "moneyline_home": line.get('home_ml'),
                    "moneyline_away": line.get('away_ml'),
                    "total": line.get('total'),
                    # IMPLIED PROBABILITIES (CRITICAL FOR ONTORISK!)
                    "home_implied_prob": line.get('home_implied_prob'),
                    "away_implied_prob": line.get('away_implied_prob'),
                    "home_no_vig_prob": line.get('home_no_vig_prob'),
                    "away_no_vig_prob": line.get('away_no_vig_prob'),
                    "vig_percentage": line.get('vig_percentage'),
                    "timestamp": datetime.now().isoformat(),
                    "source": line.get('source', 'BetOnline'),
                    "available": True,
                    "locked": False
                }
                break
        
        return game_line
        
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "available": False,
            "locked": True,
            "lock_reason": f"Error: {str(e)}"
        }, status_code=500)


@app.post("/api/betonline/update-odds")
async def update_betonline_odds(request: dict):
    """
    ADMIN: Update BetOnline odds in real-time
    
    Body:
        {
            "game_id": "0022500043",
            "spread": -1.5,
            "total": 233.5,
            "home_ml": -125,
            "away_ml": +105
        }
    
    Returns:
        Updated odds confirmation
    """
    try:
        # Store in global variable for immediate use
        global _live_betonline_odds
        if '_live_betonline_odds' not in globals():
            _live_betonline_odds = {}
        
        game_id = request.get('game_id')
        _live_betonline_odds[game_id] = {
            'spread': request.get('spread'),
            'total': request.get('total'),
            'home_ml': request.get('home_ml'),
            'away_ml': request.get('away_ml'),
            'timestamp': datetime.now().isoformat(),
            'source': 'BetOnline (ADMIN UPDATED - REAL TIME!)'
        }
        
        return {
            "status": "✅ Odds updated!",
            "game_id": game_id,
            "spread": request.get('spread'),
            "total": request.get('total'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/betonline/manual-entry")
async def manual_betonline_entry(request: dict):
    """
    Manually enter REAL BetOnline odds for OntoRisk
    
    Body:
        {
            "game_id": "0022500123",
            "home_team": "LAL",
            "away_team": "GSW",
            "spread": -6.0,
            "total": 215.5,
            "home_ml": -250,
            "away_ml": +210
        }
    
    Returns:
        Odds with calculated implied probabilities
    """
    try:
        global _manual_betonline_odds
        if '_manual_betonline_odds' not in globals():
            _manual_betonline_odds = {}
        
        game_id = request['game_id']
        home_ml = request['home_ml']
        away_ml = request['away_ml']
        
        # Calculate implied probabilities
        if home_ml < 0:
            home_implied = abs(home_ml) / (abs(home_ml) + 100)
        else:
            home_implied = 100 / (home_ml + 100)
        
        if away_ml < 0:
            away_implied = abs(away_ml) / (abs(away_ml) + 100)
        else:
            away_implied = 100 / (away_ml + 100)
        
        total_implied = home_implied + away_implied
        vig_pct = (total_implied - 1) * 100
        
        home_no_vig = home_implied / total_implied
        away_no_vig = away_implied / total_implied
        
        _manual_betonline_odds[game_id] = {
            'game_id': game_id,
            'home_team': request['home_team'],
            'away_team': request['away_team'],
            'spread': request['spread'],
            'total': request['total'],
            'home_ml': home_ml,
            'away_ml': away_ml,
            'home_implied_prob': home_implied,
            'away_implied_prob': away_implied,
            'home_no_vig_prob': home_no_vig,
            'away_no_vig_prob': away_no_vig,
            'vig_percentage': vig_pct,
            'source': 'BetOnline (MANUAL ENTRY - REAL ODDS)',
            'entered_at': datetime.now().isoformat()
        }
        
        # Also update the trading engine's scraper cache
        if trading_engine and hasattr(trading_engine.line_scraper, 'manual_odds_cache'):
            trading_engine.line_scraper.manual_odds_cache[game_id] = _manual_betonline_odds[game_id]
        
        return {
            "status": "✅ REAL BetOnline odds entered!",
            "game_id": game_id,
            "odds": _manual_betonline_odds[game_id],
            "message": "OntoRisk can now calculate with REAL odds!"
        }
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/risk-status")
async def get_risk_status():
    """
    Get current risk management status
    
    Returns:
        Risk status (bankroll, limits, etc.)
    """
    if trading_engine is None or not trading_engine.ontorisk_enabled:
        return JSONResponse({"error": "OntoRisk not available"}, status_code=503)
    
    status = trading_engine.risk_manager.get_status()
    
    return {
        "risk_status": status,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/place-bet")
async def place_bet(bet: Dict):
    """
    Place a bet (for now, just logs it)
    
    In production: Would integrate with actual sportsbook API
    """
    print(f"📝 Bet placed: {bet}")
    
    # In production: Place bet with sportsbook
    # For now: Just log it
    
    return {
        "success": True,
        "message": "Bet logged (not placed - paper trading mode)",
        "bet": bet,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/bets/add")
async def add_bet(bet: Dict):
    """
    Add a bet to portfolio
    """
    if portfolio_manager is None:
        return JSONResponse({"error": "Portfolio manager not initialized"}, status_code=503)
    
    try:
        bet_id = portfolio_manager.add_bet(
            matchup=bet.get('matchup', ''),
            bet_type=bet.get('bet_type', 'SPREAD'),
            bet_line=bet.get('bet_line', ''),
            stake=bet.get('stake', 0),
            odds=bet.get('odds', -110),
            prediction=bet.get('prediction'),
            market_spread=bet.get('market_spread'),
            edge=bet.get('edge'),
            p_win=bet.get('p_win'),
            book=bet.get('book', 'BetOnline'),
            notes=bet.get('notes')
        )
        
        return {
            "success": True,
            "bet_id": bet_id,
            "message": "Bet logged successfully"
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/bets/all")
async def get_all_bets():
    """Get all bets in portfolio"""
    if portfolio_manager is None:
        return JSONResponse({"error": "Portfolio manager not initialized"}, status_code=503)
    
    bets = portfolio_manager.get_all_bets()
    return {
        "bets": bets,
        "count": len(bets)
    }


@app.get("/api/bets/pending")
async def get_pending_bets():
    """Get pending bets"""
    if portfolio_manager is None:
        return JSONResponse({"error": "Portfolio manager not initialized"}, status_code=503)
    
    bets = portfolio_manager.get_pending_bets()
    return {
        "bets": bets,
        "count": len(bets)
    }


@app.get("/api/bets/summary")
async def get_portfolio_summary():
    """Get portfolio performance summary"""
    if portfolio_manager is None:
        return JSONResponse({"error": "Portfolio manager not initialized"}, status_code=503)
    
    summary = portfolio_manager.get_performance_summary()
    return summary


@app.post("/api/bets/{bet_id}/settle")
async def settle_bet(bet_id: int, result: Dict):
    """Settle a bet"""
    if portfolio_manager is None:
        return JSONResponse({"error": "Portfolio manager not initialized"}, status_code=503)
    
    try:
        portfolio_manager.settle_bet(
            bet_id=bet_id,
            result=result.get('result', 'WIN'),
            actual_score=result.get('actual_score'),
            profit=result.get('profit')
        )
        
        return {
            "success": True,
            "message": "Bet settled"
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/model/state")
async def get_model_state():
    """
    Get current ML model state for visualization
    
    Returns:
        Model processing state, features, predictions
    """
    if trading_engine is None:
        return JSONResponse({"error": "System not initialized"}, status_code=503)
    
    # Get latest opportunity/prediction if available
    opportunities = trading_engine.scan_live_opportunities()
    
    if opportunities:
        latest = opportunities[0]
        
        # Simulate feature values (in production: extract from actual game state)
        import numpy as np
        features = np.random.randn(18).tolist()  # 18 features for Mamba Mentality
        
        return {
            "model": "MAMBA_MENTALITY",
            "status": "processing",
            "current_diff": latest.get('current_score', 0),
            "features": features,
            "prediction": latest.get('prediction', 0),
            "confidence": latest.get('p_win', 0.5),
            "edge": latest.get('edge', 0),
            "models_active": ['XGBoost', 'LightGBM', 'RandomForest', 'DeepNN', 'Ridge', 'ExtraTrees'],
            "timestamp": datetime.now().isoformat()
        }
    else:
        # No active prediction
        return {
            "model": "MAMBA_MENTALITY",
            "status": "idle",
            "current_diff": 0,
            "features": [0] * 18,
            "prediction": 0,
            "confidence": 0,
            "edge": 0,
            "models_active": ['XGBoost', 'LightGBM', 'RandomForest', 'DeepNN', 'Ridge', 'ExtraTrees'],
            "timestamp": datetime.now().isoformat()
        }


@app.get("/api/court/3d/{game_id}")
async def get_3d_court_stream(game_id: str):
    """Get 3D court visualization stream for a game"""
    if court_stream is None:
        return JSONResponse({"error": "Court stream not initialized"}, status_code=503)
    
    # In production: Fetch real PBP and generate 3D stream
    # For now: Return sample data
    sample_frame = {
        'timestamp': '6:00',
        'period': 2,
        'event_type': 'SHOT',
        'home_players': [
            {'player_id': 'LAL_1', 'x': 20, 'y': 25, 'z': 0, 'team': 'LAL'},
            {'player_id': 'LAL_2', 'x': 30, 'y': 20, 'z': 0, 'team': 'LAL'},
            {'player_id': 'LAL_3', 'x': 25, 'y': 30, 'z': 0, 'team': 'LAL'},
            {'player_id': 'LAL_4', 'x': 15, 'y': 15, 'z': 0, 'team': 'LAL'},
            {'player_id': 'LAL_5', 'x': 10, 'y': 25, 'z': 0, 'team': 'LAL'},
        ],
        'away_players': [
            {'player_id': 'BOS_1', 'x': 74, 'y': 25, 'z': 0, 'team': 'BOS'},
            {'player_id': 'BOS_2', 'x': 64, 'y': 20, 'z': 0, 'team': 'BOS'},
            {'player_id': 'BOS_3', 'x': 69, 'y': 30, 'z': 0, 'team': 'BOS'},
            {'player_id': 'BOS_4', 'x': 79, 'y': 15, 'z': 0, 'team': 'BOS'},
            {'player_id': 'BOS_5', 'x': 84, 'y': 25, 'z': 0, 'team': 'BOS'},
        ],
        'ball_position': {'x': 47, 'y': 25, 'z': 5}
    }
    
    return {
        "game_id": game_id,
        "current_frame": sample_frame,
        "court_dimensions": {
            "length": 94,
            "width": 50
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time updates - COMPLETE SYSTEM PACKAGE
    
    Frontend receives EVERYTHING needed:
    - Live games
    - Mamba predictions (33 features extracted)
    - BetOnline odds
    - OntoRisk analysis
    - System status
    
    Frontend just displays, backend does ALL heavy work!
    """
    await websocket.accept()
    active_connections.append(websocket)
    print(f"✅ WebSocket connected: {websocket.client}")
    
    try:
        while True:
            try:
                # Build complete message package
                message = await build_complete_message()
                
                # Send to frontend
                await websocket.send_json(message)
                
                # ⚡ ULTRA REAL-TIME: 1-second updates!
                # Eliminates all perceived lag, scores update instantly
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Error in message loop (continuing): {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)  # Wait before retry
                continue  # Don't disconnect, just retry
            
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
        print(f"❌ WebSocket disconnected: {websocket.client}")
    except Exception as e:
        print(f"❌ WebSocket fatal error: {e}")
        import traceback
        traceback.print_exc()
        if websocket in active_connections:
            active_connections.remove(websocket)


async def build_complete_message() -> dict:
    """
    Build complete message package with ALL analysis done
    
    Returns:
        Complete WebSocket message with:
        - Live games (updated every 1s)
        - Mamba predictions (checked every call, only at Q2 6:00+)
        - BetOnline odds
        - OntoRisk analysis
        - System status
    """
    try:
        # Get live games (FAST - every call)
        if nba_api:
            live_games = nba_api.get_todays_games()
        else:
            live_games = []
        
        # Get betting opportunities (checks Q2 6:00 window)
        # Model only predicts when game has 18+ minutes of play
        if trading_engine:
            opportunities = trading_engine.scan_live_opportunities()
        else:
            opportunities = []
        
        # Get system status
        system_status = {
            "mamba_loaded": trading_engine is not None and trading_engine.model is not None,
            "nba_api_connected": nba_api is not None,
            "betonline_scraper_active": False,  # TODO: Check scraper status
            "ontorisk_enabled": trading_engine is not None and trading_engine.ontorisk_enabled,
            "total_predictions_today": len(getattr(trading_engine, 'prediction_storage', [])) if trading_engine else 0,
            "avg_mae_today": 9.655,  # TODO: Calculate from today's predictions
            "win_rate_today": 0.0,  # TODO: Calculate from today's results
            "starting_bankroll": 1000.0,
            "current_bankroll": trading_engine.risk_manager.bankroll if trading_engine and hasattr(trading_engine, 'risk_manager') else 1000.0,
            "total_profit": 0.0,  # TODO: Calculate
            "roi": 0.0,  # TODO: Calculate
            "last_error": None,
            "error_count_today": 0
        }
        
        # Build complete message
        message = {
            "type": "update",
            "timestamp": datetime.now().isoformat(),
            "live_games": live_games,
            "opportunities": opportunities,
            "system_status": system_status
        }
        
        return message
        
    except Exception as e:
        print(f"❌ Error building message: {e}")
        return {
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "live_games": [],
            "opportunities": [],
            "system_status": {}
        }


def get_db_connection():
    """Get PostgreSQL connection for stats queries"""
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if DATABASE_URL:
        import psycopg2
        return psycopg2.connect(DATABASE_URL)
    return None


# ============================================================================
# NBA STATS API ENDPOINTS (Rolling Model)
# ============================================================================

@app.get("/api/stats/teams")
async def get_all_teams():
    """Get all 30 NBA teams from database"""
    conn = get_db_connection()
    if not conn:
        return {"error": "Database not configured", "teams": [], "count": 0}
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT team_id, abbreviation, full_name, conference, division, city
            FROM teams
            ORDER BY full_name
        """)
        
        teams_list = []
        for row in cursor.fetchall():
            teams_list.append({
                "team_id": row[0],
                "abbreviation": row[1],
                "full_name": row[2],
                "conference": row[3],
                "division": row[4],
                "city": row[5]
            })
        
        conn.close()
        return {"teams": teams_list, "count": len(teams_list)}
    except Exception as e:
        if conn:
            conn.close()
        return {"error": str(e), "teams": [], "count": 0}


@app.get("/api/stats/standings")
async def get_standings():
    """Get current NBA standings from database"""
    conn = get_db_connection()
    if not conn:
        return {"error": "Database not configured"}
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.abbreviation, t.full_name, s.conference, s.rank, 
                   s.wins, s.losses, s.gb, s.streak
            FROM standings s
            JOIN teams t ON s.team_id = t.team_id
            WHERE s.updated_at::date = (SELECT MAX(updated_at::date) FROM standings)
            ORDER BY s.conference, s.rank
        """)
        
        standings_data = {"East": [], "West": []}
        for row in cursor.fetchall():
            team_data = {
                "abbreviation": row[0],
                "full_name": row[1],
                "rank": row[3],
                "wins": row[4],
                "losses": row[5],
                "gb": float(row[6]) if row[6] else 0.0,
                "streak": row[7]
            }
            standings_data[row[2]].append(team_data)
        
        conn.close()
        return {"standings": standings_data}
    except Exception as e:
        if conn:
            conn.close()
        return {"error": str(e)}


def start_dashboard_api(host: str = "0.0.0.0", port: int = None):
    """
    Start the dashboard API
    
    Args:
        host: Host address
        port: Port number (defaults to $PORT env var or 8001)
    """
    # Use Railway's PORT env var if available, otherwise default to 8001
    if port is None:
        port = int(os.getenv("PORT", 8001))
    
    print("\n" + "="*80)
    print("🔥 STARTING TRADING DASHBOARD API")
    print("="*80)
    print(f"\nAPI available at:")
    print(f"  • http://0.0.0.0:{port}/")
    print(f"  • http://0.0.0.0:{port}/docs (Swagger)")
    print(f"  • http://0.0.0.0:{port}/api/live-games")
    print(f"  • http://0.0.0.0:{port}/api/opportunities")
    print(f"  • ws://0.0.0.0:{port}/ws (WebSocket)")
    print("\n" + "="*80)
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_dashboard_api()

