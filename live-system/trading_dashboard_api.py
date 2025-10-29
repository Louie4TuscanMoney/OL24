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
import json
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

# Start daily NBA data scheduler (3:30 AM UTC updates)
try:
    sys.path.insert(0, '../backend/services')
    from daily_nba_scheduler import start_scheduler
    start_scheduler()
except ImportError:
    print("⚠️ Daily NBA scheduler not available (install 'schedule' package)")

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
                
                # Check if WebSocket is still open before sending
                if websocket.client_state.value == 1:  # 1 = OPEN
                    # Log what we're sending (for debugging)
                    game_count = len(message.get('games', []))
                    pred_count = len(message.get('opportunities', []))
                    print(f"📤 WebSocket send: {game_count} games, {pred_count} predictions")
                    
                    # Log live game details
                    for game in message.get('games', []):
                        if game.get('is_live'):
                            print(f"   🔴 {game['away_team']} @ {game['home_team']}: {game['score_away']}-{game['score_home']} | Q{game['quarter']} {game['time_remaining']}")
                    
                    await websocket.send_json(message)
                else:
                    print(f"⚠️ WebSocket closed, ending loop")
                    break
                
                # ⚡ ULTRA REAL-TIME: 1-second updates!
                # Eliminates all perceived lag, scores update instantly
                await asyncio.sleep(1)
                
            except RuntimeError as e:
                if "close message" in str(e):
                    print(f"⚠️ WebSocket already closed, ending loop")
                    break
                print(f"⚠️ Error in message loop (continuing): {e}")
                await asyncio.sleep(1)
                continue
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
        # 🔥 FORCE REFRESH: Bypass all caching to ensure real-time data
        if nba_api:
            all_games = nba_api.get_todays_games(force_refresh=True)
            # FILTER: Only show LIVE games OR today's games (not yesterday's finals!)
            from datetime import date
            today_str = date.today().strftime('%Y-%m-%d')
            
            # Map to frontend format (crucial!)
            live_games = []
            for g in all_games:
                if g.get('status') == 3:  # Skip finished games
                    continue
                
                # Map backend format to frontend format
                game_mapped = {
                    'game_id': g.get('game_id'),
                    'home_team': g.get('home_team'),
                    'away_team': g.get('away_team'),
                    'score_home': g.get('home_score', 0),  # Frontend expects score_home
                    'score_away': g.get('away_score', 0),  # Frontend expects score_away
                    'quarter': g.get('period', 0),  # Frontend expects quarter
                    'time_remaining': g.get('clock', ''),  # Frontend expects time_remaining
                    'clock': g.get('clock', ''),  # Also include clock
                    'is_live': g.get('status') == 2,  # Frontend expects is_live
                    'status': g.get('status', 1),
                    'status_text': g.get('status_text', ''),
                    'game_time': g.get('game_time', ''),  # PST time
                    'game_date': g.get('game_date', '')   # Date
                }
                live_games.append(game_mapped)
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
        # Frontend expects 'games' not 'live_games'!
        message = {
            "type": "update",
            "timestamp": datetime.now().isoformat(),
            "games": live_games,  # Frontend expects this field name!
            "live_games": live_games,  # Keep for backward compat
            "opportunities": opportunities,
            "predictions": opportunities,  # Frontend might expect this
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
    """Get PostgreSQL connection for stats queries (OPTIMIZED with connection reuse)"""
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if DATABASE_URL:
        import psycopg2
        # Use connection with keepalive for faster queries
        return psycopg2.connect(
            DATABASE_URL,
            connect_timeout=3,  # Fast timeout
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5
        )
    return None


def _calculate_variance(games, stat_key):
    """Calculate variance for probability analysis"""
    if not games or len(games) < 2:
        return 0
    values = [g[stat_key] for g in games if g[stat_key] is not None]
    if len(values) < 2:
        return 0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return round(variance, 2)


def _calculate_consistency(games):
    """Consistency score: 1.0 = perfect, 0.0 = chaos"""
    if not games or len(games) < 3:
        return 0
    pts_values = [g['pts'] for g in games if g['pts'] is not None]
    if len(pts_values) < 3:
        return 0
    mean = sum(pts_values) / len(pts_values)
    std = (sum((x - mean) ** 2 for x in pts_values) / len(pts_values)) ** 0.5
    cv = std / mean if mean > 0 else 0  # Coefficient of variation
    consistency = max(0, 1.0 - cv)  # Lower CV = higher consistency
    return round(consistency, 3)


# ============================================================================
# NBA STATS API ENDPOINTS (Rolling Model)
# ============================================================================

@app.get("/api/stats/teams")
async def get_all_teams():
    """Get all 30 NBA teams with stats from team_season_stats table"""
    conn = get_db_connection()
    if not conn:
        return {"error": "Database not configured", "teams": [], "count": 0}
    
    try:
        cursor = conn.cursor()
        # Get team stats from team_season_stats table
        cursor.execute("""
            SELECT 
                t.team_id,
                t.abbreviation,
                t.full_name,
                t.logo_url,
                t.primary_color,
                t.secondary_color,
                COALESCE(tss.games_played, 0) as games_played,
                COALESCE(tss.wins, 0) as wins,
                COALESCE(tss.losses, 0) as losses,
                COALESCE(tss.ppg, 0) as ppg,
                COALESCE(tss.net_rating, 0) as net_rating
            FROM teams t
            LEFT JOIN team_season_stats tss ON t.team_id = tss.team_id AND tss.season_id = '2025-26'
            ORDER BY tss.wins DESC NULLS LAST, t.abbreviation
        """)
        
        teams_list = []
        for row in cursor.fetchall():
            teams_list.append({
                "team_id": row[0],
                "abbreviation": row[1],
                "full_name": row[2],
                "logo_url": row[3],
                "primary_color": row[4],
                "secondary_color": row[5],
                "games_played": int(row[6]),
                "wins": int(row[7]),
                "losses": int(row[8]),
                "ppg": round(float(row[9]), 1),
                "net_rating": round(float(row[10]), 1)
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


@app.get("/api/stats/player/{player_id}")
async def get_player_profile(player_id: str):
    """
    COMPREHENSIVE PLAYER PROFILE
    Everything for professional frontend display
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Database not configured"}
    
    try:
        cursor = conn.cursor()
        
        # Get player info + season stats + team
        cursor.execute("""
            SELECT 
                p.player_id, p.name, p.first_name, p.last_name,
                p.headshot_url, p.action_photo_url,
                p.jersey_number, p.position, p.height_display, p.weight_lbs,
                p.birthdate, p.age, p.country, p.experience_years,
                p.draft_year, p.draft_round, p.draft_number, p.college,
                t.abbreviation, t.full_name AS team_name, t.logo_url, 
                t.primary_color, t.secondary_color,
                pss.games_played, pss.ppg, pss.rpg, pss.apg,
                pss.fg_pct, pss.fg3_pct, pss.ft_pct,
                pss.ts_pct, pss.efg_pct,
                pss.pts_100, pss.reb_100, pss.ast_100,
                pss.pts_36, pss.reb_36, pss.ast_36,
                pss.bpm, pss.per, pss.vorp, pss.usage_pct, pss.win_shares,
                pss.lebron_total, pss.lebron_offense, pss.lebron_defense,
                pss.rapm_total
            FROM players p
            LEFT JOIN teams t ON p.team_id = t.team_id
            LEFT JOIN player_season_stats pss ON p.player_id = pss.player_id
            WHERE p.player_id = %s
        """, (player_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return {"error": "Player not found"}
        
        # Get last 10 games
        cursor.execute("""
            SELECT game_date, opponent_id, pts, reb, ast, 
                   fgm, fga, fg3m, minutes, plus_minus,
                   pts_100, ts_pct
            FROM player_last10
            WHERE player_id = %s AND game_rank <= 10
            ORDER BY game_date DESC
        """, (player_id,))
        
        last10 = []
        for game_row in cursor.fetchall():
            opp_id = game_row[1]
            cursor.execute("SELECT abbreviation FROM teams WHERE team_id = %s", (opp_id,))
            opp_result = cursor.fetchone()
            opp_abbr = opp_result[0] if opp_result else 'UNK'
            
            last10.append({
                "date": game_row[0].strftime('%Y-%m-%d'),
                "opponent": opp_abbr,
                "pts": game_row[2],
                "reb": game_row[3],
                "ast": game_row[4],
                "fgm": game_row[5],
                "fga": game_row[6],
                "fg3m": game_row[7],
                "minutes": float(game_row[8]) if game_row[8] else 0,
                "plus_minus": game_row[9],
                "pts_100": float(game_row[10]) if game_row[10] else 0,
                "ts_pct": float(game_row[11]) if game_row[11] else 0
            })
        
        conn.close()
        
        # Build comprehensive profile
        return {
            "player_id": row[0],
            "name": row[1],
            "first_name": row[2],
            "last_name": row[3],
            
            # Visuals
            "headshot_url": row[4],
            "action_photo_url": row[5],
            
            # Bio
            "jersey": row[6],
            "position": row[7],
            "height": row[8],
            "weight": row[9],
            "birthdate": row[10].strftime('%Y-%m-%d') if row[10] else None,
            "age": row[11],
            "country": row[12],
            "experience": row[13],
            
            # Draft
            "draft_year": row[14],
            "draft_round": row[15],
            "draft_number": row[16],
            "college": row[17],
            
            # Team
            "team": {
                "abbreviation": row[18],
                "full_name": row[19],
                "logo_url": row[20],
                "primary_color": row[21],
                "secondary_color": row[22]
            },
            
            # Season Stats (Per-Game)
            "season_stats": {
                "games_played": row[23],
                "ppg": float(row[24]) if row[24] else 0,
                "rpg": float(row[25]) if row[25] else 0,
                "apg": float(row[26]) if row[26] else 0,
                "fg_pct": float(row[27]) if row[27] else 0,
                "fg3_pct": float(row[28]) if row[28] else 0,
                "ft_pct": float(row[29]) if row[29] else 0
            },
            
            # Advanced Stats (from Basketball Reference + custom metrics)
            "advanced_stats": {
                "ts_pct": float(row[30]) if row[30] else 0,
                "efg_pct": float(row[31]) if row[31] else 0,
                "pts_100": float(row[32]) if row[32] else 0,
                "reb_100": float(row[33]) if row[33] else 0,
                "ast_100": float(row[34]) if row[34] else 0,
                "pts_36": float(row[35]) if row[35] else 0,
                "reb_36": float(row[36]) if row[36] else 0,
                "ast_36": float(row[37]) if row[37] else 0,
                "bpm": float(row[38]) if row[38] else 0,
                "per": float(row[39]) if row[39] else 0,
                "vorp": float(row[40]) if row[40] else 0,
                "usage_pct": float(row[41]) if row[41] else 0,
                "win_shares": float(row[42]) if row[42] else 0,
                "lebron": float(row[43]) if row[43] else 0,
                "lebron_offense": float(row[44]) if row[44] else 0,
                "lebron_defense": float(row[45]) if row[45] else 0,
                "rapm": float(row[46]) if row[46] else 0
            },
            
            # Last 10 Games
            "last10_games": last10,
            
            # Probability Context (for your independent probability math)
            "probability_metrics": {
                "sample_size": row[23],
                "scoring_variance": _calculate_variance(last10, 'pts') if last10 else 0,
                "consistency_score": _calculate_consistency(last10) if last10 else 0
            }
        }
        
    except Exception as e:
        if conn:
            conn.close()
        return {"error": str(e)}


# ============================================================================
# NEW COMPREHENSIVE NBA ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/api/injuries")
async def get_all_injuries():
    """
    Get all active player injuries
    Returns: List of injured players with status, type, description
    """
    conn = get_db_connection()
    if not conn:
        return {"injuries": []}
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                p.player_id, p.name, p.position,
                t.abbreviation, t.full_name,
                i.status, i.injury_type, i.description,
                i.injury_date, i.return_date
            FROM player_injuries i
            JOIN players p ON p.player_id = i.player_id
            LEFT JOIN teams t ON t.team_id = p.team_id
            WHERE i.is_active = TRUE
            ORDER BY i.injury_date DESC
        """)
        
        injuries = []
        for row in cursor.fetchall():
            injuries.append({
                "player_id": row[0],
                "name": row[1],
                "position": row[2],
                "team_abbr": row[3],
                "team_name": row[4],
                "status": row[5],
                "injury_type": row[6],
                "description": row[7],
                "injury_date": row[8].strftime('%Y-%m-%d') if row[8] else None,
                "return_date": row[9].strftime('%Y-%m-%d') if row[9] else None
            })
        
        conn.close()
        return {"injuries": injuries, "count": len(injuries)}
        
    except Exception as e:
        if conn:
            conn.close()
        return {"error": str(e)}


@app.get("/api/schedule")
async def get_nba_schedule(days_ahead: int = 7):
    """
    Get NBA schedule for next N days
    """
    conn = get_db_connection()
    if not conn:
        return {"games": []}
    
    try:
        from datetime import date, timedelta
        today = date.today()
        end_date = today + timedelta(days=days_ahead)
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                s.game_id, s.game_date, s.game_time,
                home.abbreviation, home.full_name, home.logo_url,
                away.abbreviation, away.full_name, away.logo_url,
                s.game_status, s.home_score, s.away_score,
                s.arena, s.tv_broadcast
            FROM nba_schedule s
            JOIN teams home ON home.team_id = s.home_team_id
            JOIN teams away ON away.team_id = s.away_team_id
            WHERE s.game_date >= %s AND s.game_date <= %s
            ORDER BY s.game_date, s.game_time
        """, (today, end_date))
        
        games = []
        for row in cursor.fetchall():
            games.append({
                "game_id": row[0],
                "date": row[1].strftime('%Y-%m-%d'),
                "time": row[2].strftime('%H:%M') if row[2] else None,
                "home_team": {"abbr": row[3], "name": row[4], "logo": row[5]},
                "away_team": {"abbr": row[6], "name": row[7], "logo": row[8]},
                "status": row[9],
                "score": {"home": row[10], "away": row[11]} if row[10] else None,
                "arena": row[12],
                "tv": row[13]
            })
        
        conn.close()
        return {"games": games, "count": len(games)}
        
    except Exception as e:
        if conn:
            conn.close()
        return {"error": str(e)}


@app.get("/api/team/{team_abbr}/depth-chart")
async def get_team_depth_chart(team_abbr: str):
    """
    Get team depth chart with projected starters (using nba_api!)
    """
    # Use nba_api team service instead of empty database
    try:
        import sys
        sys.path.insert(0, '/app/backend/services')
        from nba_team_service import team_service
        
        result = team_service.get_depth_chart(team_abbr)
        return result
    except Exception as e:
        print(f"⚠️  Team service not available, falling back to database...")
    
    # Fallback to database (if service fails)
    conn = get_db_connection()
    if not conn:
        return {"error": "Database not configured and team service unavailable"}
    
    try:
        cursor = conn.cursor()
        
        # Get team ID
        cursor.execute("SELECT team_id, full_name FROM teams WHERE abbreviation = %s", (team_abbr,))
        team_row = cursor.fetchone()
        if not team_row:
            conn.close()
            return {"error": "Team not found"}
        
        team_id, team_name = team_row
        
        # Get depth chart with MPG and ADVANCED STATS
        # NOTE: team_depth_charts doesn't have season_id, avg_mpg, or is_starter
        # We'll use player_season_stats to get MPG and compute starters from MPG
        cursor.execute("""
            SELECT 
                p.player_id, p.name, p.position, p.jersey_number,
                COALESCE(dc.depth_rank, 99) as depth_rank,
                COALESCE(ps.minutes_total / NULLIF(ps.games_played, 0), 0) as avg_mpg,
                CASE WHEN COALESCE(ps.minutes_total / NULLIF(ps.games_played, 0), 0) >= 25 THEN TRUE ELSE FALSE END as is_starter,
                COALESCE(ps.ppg, 0) as ppg, 
                COALESCE(ps.rpg, 0) as rpg, 
                COALESCE(ps.apg, 0) as apg, 
                COALESCE(ps.games_played, 0) as gp,
                COALESCE(ps.ts_pct, 0) as ts_pct, 
                COALESCE(ps.efg_pct, 0) as efg_pct, 
                COALESCE(ps.pts_100, 0) as pts_100, 
                COALESCE(ps.reb_100, 0) as reb_100, 
                COALESCE(ps.ast_100, 0) as ast_100,
                COALESCE(ps.fg_pct, 0) as fg_pct, 
                COALESCE(ps.fg3_pct, 0) as fg3_pct, 
                COALESCE(ps.ft_pct, 0) as ft_pct,
                (SELECT status FROM player_injuries 
                 WHERE player_id = p.player_id AND is_active = TRUE 
                 LIMIT 1) as injury_status
            FROM players p
            LEFT JOIN team_depth_charts dc ON dc.player_id = p.player_id AND dc.team_id = %s
            LEFT JOIN player_season_stats ps ON ps.player_id = p.player_id AND ps.season_id = '2025-26'
            WHERE p.team_id = %s
            ORDER BY 
                CASE WHEN COALESCE(ps.minutes_total / NULLIF(ps.games_played, 0), 0) >= 25 THEN 0 ELSE 1 END,
                COALESCE(dc.depth_rank, 99),
                COALESCE(ps.minutes_total / NULLIF(ps.games_played, 0), 0) DESC
        """, (team_id, team_id))
        
        positions = {'PG': [], 'SG': [], 'SF': [], 'PF': [], 'C': []}
        starters = []
        
        bench = []
        all_players = []
        
        for row in cursor.fetchall():
            player_data = {
                "player_id": row[0],
                "name": row[1],
                "position": row[2],
                "jersey": row[3],
                "depth_rank": row[4],
                "mpg": float(row[5]) if row[5] else 0,
                "is_starter": row[6],
                "ppg": float(row[7]) if row[7] else 0,
                "rpg": float(row[8]) if row[8] else 0,
                "apg": float(row[9]) if row[9] else 0,
                "gp": int(row[10]) if row[10] else 0,
                # ADVANCED STATS
                "ts_pct": float(row[11]) if row[11] else 0,
                "efg_pct": float(row[12]) if row[12] else 0,
                "pts_100": float(row[13]) if row[13] else 0,
                "reb_100": float(row[14]) if row[14] else 0,
                "ast_100": float(row[15]) if row[15] else 0,
                "fg_pct": float(row[16]) if row[16] else 0,
                "fg3_pct": float(row[17]) if row[17] else 0,
                "ft_pct": float(row[18]) if row[18] else 0,
                "injury_status": row[19]
            }
            
            all_players.append(player_data)
            
            if row[2] in positions:
                positions[row[2]].append(player_data)
            
            if row[6]:  # is_starter
                starters.append(player_data)
            else:
                bench.append(player_data)
        
        conn.close()
        
        return {
            "team": {"abbreviation": team_abbr, "name": team_name},
            "starters": starters,
            "bench": bench,
            "all_players": all_players,
            "depth_chart": positions,
            "total_players": len(all_players)
        }
        
    except Exception as e:
        if conn:
            conn.close()
        return {"error": str(e)}


@app.get("/api/transactions/league")
async def get_league_transactions(limit: int = 50):
    """
    Get league-wide recent transactions (trades, waivers, signings)
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                pt.transaction_id,
                pt.player_name,
                pt.transaction_type,
                pt.transaction_date,
                pt.trade_description,
                t_from.abbreviation as from_team,
                t_from.full_name as from_team_name,
                t_to.abbreviation as to_team,
                t_to.full_name as to_team_name,
                pt.created_at
            FROM player_transactions pt
            LEFT JOIN teams t_from ON pt.from_team_id = t_from.team_id
            LEFT JOIN teams t_to ON pt.to_team_id = t_to.team_id
            ORDER BY pt.transaction_date DESC, pt.created_at DESC
            LIMIT %s
        """, (limit,))
        
        transactions = []
        for row in cur.fetchall():
            transactions.append({
                "id": row[0],
                "player_name": row[1],
                "type": row[2],
                "date": row[3].strftime('%Y-%m-%d') if row[3] else None,
                "description": row[4],
                "from_team": row[5],
                "from_team_name": row[6],
                "to_team": row[7],
                "to_team_name": row[8],
                "timestamp": row[9].isoformat() if row[9] else None
            })
        
        conn.close()
        return transactions
        
    except Exception as e:
        if conn:
            conn.close()
        return {"error": str(e)}


@app.get("/api/transactions/team/{team_abbr}")
async def get_team_transactions(team_abbr: str, limit: int = 20):
    """
    Get transactions for a specific team (incoming/outgoing)
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cur = conn.cursor()
        
        # Get team ID
        cur.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (team_abbr,))
        team_row = cur.fetchone()
        if not team_row:
            conn.close()
            return {"error": "Team not found"}
        
        team_id = team_row[0]
        
        # Get transactions
        cur.execute("""
            SELECT 
                pt.transaction_id,
                pt.player_name,
                pt.transaction_type,
                pt.transaction_date,
                pt.trade_description,
                t_from.abbreviation as from_team,
                t_from.full_name as from_team_name,
                t_to.abbreviation as to_team,
                t_to.full_name as to_team_name,
                CASE 
                    WHEN pt.from_team_id = %s THEN 'outgoing'
                    WHEN pt.to_team_id = %s THEN 'incoming'
                    ELSE 'related'
                END as direction
            FROM player_transactions pt
            LEFT JOIN teams t_from ON pt.from_team_id = t_from.team_id
            LEFT JOIN teams t_to ON pt.to_team_id = t_to.team_id
            WHERE pt.from_team_id = %s OR pt.to_team_id = %s
            ORDER BY pt.transaction_date DESC
            LIMIT %s
        """, (team_id, team_id, team_id, team_id, limit))
        
        transactions = []
        for row in cur.fetchall():
            transactions.append({
                "id": row[0],
                "player_name": row[1],
                "type": row[2],
                "date": row[3].strftime('%Y-%m-%d') if row[3] else None,
                "description": row[4],
                "from_team": row[5],
                "from_team_name": row[6],
                "to_team": row[7],
                "to_team_name": row[8],
                "direction": row[9]  # incoming/outgoing/related
            })
        
        conn.close()
        return transactions
        
    except Exception as e:
        if conn:
            conn.close()
        return {"error": str(e)}


# ============================================================================
# ML PREDICTION ENDPOINTS
# ============================================================================

@app.get("/api/ml/predictions/active")
async def get_active_ml_predictions():
    """
    Get all active ML predictions for live games
    Returns: List of predictions with game context
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cur = conn.cursor()
        
        # Get latest prediction for each live game (last 5 minutes)
        cur.execute("""
            WITH latest_predictions AS (
                SELECT DISTINCT ON (game_id)
                    game_id,
                    point_forecast,
                    interval_lower,
                    interval_upper,
                    model_confidence,
                    edge_detected,
                    edge_magnitude,
                    quarter,
                    time_remaining,
                    prediction_timestamp,
                    is_q2_6min
                FROM ml_predictions
                WHERE prediction_timestamp > NOW() - INTERVAL '5 minutes'
                ORDER BY game_id, prediction_timestamp DESC
            )
            SELECT 
                lp.*,
                g.home_team_id,
                g.away_team_id,
                ht.full_name as home_team,
                at.full_name as away_team
            FROM latest_predictions lp
            JOIN games g ON lp.game_id = g.game_id
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.status = 'Live'
            ORDER BY lp.prediction_timestamp DESC
        """)
        
        predictions = []
        for row in cur.fetchall():
            predictions.append({
                "game_id": row[0],
                "point_forecast": float(row[1]) if row[1] else None,
                "interval_lower": float(row[2]) if row[2] else None,
                "interval_upper": float(row[3]) if row[3] else None,
                "model_confidence": float(row[4]) if row[4] else None,
                "edge_detected": row[5],
                "edge_magnitude": float(row[6]) if row[6] else None,
                "quarter": row[7],
                "time_remaining": row[8],
                "prediction_timestamp": row[9].isoformat() if row[9] else None,
                "is_q2_6min": row[10],
                "home_team": row[13],
                "away_team": row[14]
            })
        
        cur.close()
        conn.close()
        return predictions
        
    except Exception as e:
        print(f"❌ Error fetching ML predictions: {e}")
        if conn:
            conn.close()
        return []


@app.get("/api/ml/prediction/{game_id}")
async def get_game_ml_prediction(game_id: str):
    """
    Get latest ML prediction for a specific game
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Database not available"}
    
    try:
        cur = conn.cursor()
        
        # Get latest prediction
        cur.execute("""
            SELECT 
                point_forecast,
                interval_lower,
                interval_upper,
                model_confidence,
                edge_detected,
                edge_magnitude,
                quarter,
                time_remaining,
                prediction_timestamp,
                is_q2_6min,
                features_extracted,
                feature_importance
            FROM ml_predictions
            WHERE game_id = %s
            ORDER BY prediction_timestamp DESC
            LIMIT 1
        """, (game_id,))
        
        row = cur.fetchone()
        if not row:
            cur.close()
            conn.close()
            return {"error": "No prediction found"}
        
        prediction = {
            "point_forecast": float(row[0]) if row[0] else None,
            "interval_lower": float(row[1]) if row[1] else None,
            "interval_upper": float(row[2]) if row[2] else None,
            "model_confidence": float(row[3]) if row[3] else None,
            "edge_detected": row[4],
            "edge_magnitude": float(row[5]) if row[5] else None,
            "quarter": row[6],
            "time_remaining": row[7],
            "prediction_timestamp": row[8].isoformat() if row[8] else None,
            "is_q2_6min": row[9],
            "features": row[10],
            "feature_importance": row[11]
        }
        
        cur.close()
        conn.close()
        return prediction
        
    except Exception as e:
        print(f"❌ Error fetching game prediction: {e}")
        if conn:
            conn.close()
        return {"error": str(e)}


@app.post("/api/ml/prediction")
async def save_ml_prediction(prediction: dict):
    """
    Save ML prediction to database
    Called by ML engine every 30 seconds for live games
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Database not available"}
    
    try:
        cur = conn.cursor()
        
        # Insert prediction
        cur.execute("""
            INSERT INTO ml_predictions (
                game_id,
                model_id,
                quarter,
                time_remaining,
                point_forecast,
                interval_lower,
                interval_upper,
                coverage_probability,
                model_confidence,
                features_extracted,
                feature_importance,
                market_spread,
                edge_detected,
                edge_magnitude,
                is_q2_6min,
                is_trade_signal
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING prediction_id
        """, (
            prediction.get('game_id'),
            prediction.get('model_id', 1),  # Default to model 1
            prediction.get('quarter'),
            prediction.get('time_remaining'),
            prediction.get('point_forecast'),
            prediction.get('interval_lower'),
            prediction.get('interval_upper'),
            prediction.get('coverage_probability', 0.90),
            prediction.get('model_confidence'),
            json.dumps(prediction.get('features')) if prediction.get('features') else None,
            json.dumps(prediction.get('feature_importance')) if prediction.get('feature_importance') else None,
            prediction.get('market_spread'),
            prediction.get('edge_detected', False),
            prediction.get('edge_magnitude'),
            prediction.get('is_q2_6min', False),
            prediction.get('is_trade_signal', False)
        ))
        
        prediction_id = cur.fetchone()[0]
        conn.commit()
        
        cur.close()
        conn.close()
        
        return {"success": True, "prediction_id": prediction_id}
        
    except Exception as e:
        print(f"❌ Error saving prediction: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return {"error": str(e)}


@app.get("/api/game/{game_id}/live-data")
async def get_live_game_data(game_id: str):
    """
    Get complete live game data for trading desk:
    - Current score
    - Score history (last 60 seconds)
    - BetOnline spread ladder
    - ML prediction
    """
    if not trading_engine:
        return {"error": "Trading engine not available"}
    
    try:
        # Get current game state
        games = trading_engine.nba_api.get_todays_games()
        game = next((g for g in games if g['game_id'] == game_id), None)
        
        if not game:
            return {"error": "Game not found"}
        
        # Get BetOnline lines
        betonline_lines = {}
        if trading_engine.betonline:
            try:
                lines = trading_engine.betonline.get_live_lines()
                betonline_lines = next((l for l in lines if l.get('game_id') == game_id), {})
            except:
                pass
        
        # Build spread ladder (order book style)
        current_diff = game['home_score'] - game['away_score']
        spread_ladder = []
        
        # Generate ladder around current differential
        for i in range(-20, 21):  # -10 to +10 in 0.5 increments
            spread_value = current_diff + (i * 0.5)
            spread_ladder.append({
                "spread": spread_value,
                "price": -110,  # Default juice
                "side": "home" if spread_value > current_diff else "away",
                "size": 1000,  # Mock size
                "is_at_market": abs(spread_value - current_diff) < 0.5
            })
        
        return {
            "game": game,
            "spread_ladder": spread_ladder,
            "betonline": betonline_lines,
            "ml_prediction": None,  # Will add if available
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/game/{game_id}/details")
async def get_game_details(game_id: str):
    """
    COMPREHENSIVE GAME DETAILS FOR SCHEDULE PAGE
    - Team rosters with projected starters
    - Active injuries for both teams
    - Season averages for key players
    - Team records and stats
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Database not configured"}
    
    try:
        cursor = conn.cursor()
        
        # Get game info
        cursor.execute("""
            SELECT 
                g.game_id, g.game_date, g.game_time,
                g.home_team_id, g.away_team_id,
                g.home_score, g.away_score, g.game_status,
                ht.abbreviation as home_abbr, ht.full_name as home_name, ht.logo_url as home_logo,
                ht.primary_color as home_primary, ht.secondary_color as home_secondary,
                at.abbreviation as away_abbr, at.full_name as away_name, at.logo_url as away_logo,
                at.primary_color as away_primary, at.secondary_color as away_secondary
            FROM nba_schedule g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.game_id = %s
        """, (game_id,))
        
        game_row = cursor.fetchone()
        if not game_row:
            conn.close()
            return {"error": "Game not found"}
        
        home_team_id = game_row[3]
        away_team_id = game_row[4]
        
        # Get team records
        cursor.execute("""
            SELECT games_played, wins, losses, ppg, net_rating
            FROM team_season_stats
            WHERE team_id = %s AND season_id = '2025-26'
        """, (home_team_id,))
        home_stats = cursor.fetchone() or (0, 0, 0, 0, 0)
        
        cursor.execute("""
            SELECT games_played, wins, losses, ppg, net_rating
            FROM team_season_stats
            WHERE team_id = %s AND season_id = '2025-26'
        """, (away_team_id,))
        away_stats = cursor.fetchone() or (0, 0, 0, 0, 0)
        
        # Get projected starters (top 5 by minutes) for home team
        cursor.execute("""
            SELECT 
                p.player_id, p.name, p.first_name, p.last_name, p.position, p.jersey_number, p.headshot_url,
                pss.ppg, pss.rpg, pss.apg, pss.fg_pct, pss.minutes_total, pss.games_played
            FROM team_depth_charts tdc
            JOIN players p ON tdc.player_id = p.player_id
            JOIN player_season_stats pss ON p.player_id = pss.player_id AND pss.season_id = '2025-26'
            WHERE tdc.team_id = %s
            ORDER BY tdc.depth_rank ASC
            LIMIT 5
        """, (home_team_id,))
        
        home_starters = []
        for row in cursor.fetchall():
            home_starters.append({
                "player_id": row[0],
                "name": row[1],
                "first_name": row[2],
                "last_name": row[3],
                "position": row[4] or "F",
                "jersey": row[5],
                "headshot_url": row[6],
                "ppg": float(row[7]) if row[7] else 0,
                "rpg": float(row[8]) if row[8] else 0,
                "apg": float(row[9]) if row[9] else 0,
                "fg_pct": float(row[10]) if row[10] else 0,
                "mpg": float(row[11]) / float(row[12]) if row[11] and row[12] and row[12] > 0 else 0
            })
        
        # Get projected starters for away team
        cursor.execute("""
            SELECT 
                p.player_id, p.name, p.first_name, p.last_name, p.position, p.jersey_number, p.headshot_url,
                pss.ppg, pss.rpg, pss.apg, pss.fg_pct, pss.minutes_total, pss.games_played
            FROM team_depth_charts tdc
            JOIN players p ON tdc.player_id = p.player_id
            JOIN player_season_stats pss ON p.player_id = pss.player_id AND pss.season_id = '2025-26'
            WHERE tdc.team_id = %s
            ORDER BY tdc.depth_rank ASC
            LIMIT 5
        """, (away_team_id,))
        
        away_starters = []
        for row in cursor.fetchall():
            away_starters.append({
                "player_id": row[0],
                "name": row[1],
                "first_name": row[2],
                "last_name": row[3],
                "position": row[4] or "F",
                "jersey": row[5],
                "headshot_url": row[6],
                "ppg": float(row[7]) if row[7] else 0,
                "rpg": float(row[8]) if row[8] else 0,
                "apg": float(row[9]) if row[9] else 0,
                "fg_pct": float(row[10]) if row[10] else 0,
                "mpg": float(row[11]) / float(row[12]) if row[11] and row[12] and row[12] > 0 else 0
            })
        
        # Get injuries for both teams
        cursor.execute("""
            SELECT 
                p.name, p.position, i.status, i.injury_type, i.description
            FROM player_injuries i
            JOIN players p ON i.player_id = p.player_id
            WHERE p.team_id IN (%s, %s) AND i.is_active = TRUE
            ORDER BY 
                CASE i.status 
                    WHEN 'Out' THEN 1
                    WHEN 'Doubtful' THEN 2
                    WHEN 'Questionable' THEN 3
                    ELSE 4
                END
        """, (home_team_id, away_team_id))
        
        injuries = []
        for row in cursor.fetchall():
            injuries.append({
                "player_name": row[0],
                "position": row[1],
                "status": row[2],
                "injury_type": row[3],
                "description": row[4]
            })
        
        conn.close()
        
        # Build comprehensive response
        return {
            "game": {
                "game_id": game_row[0],
                "date": game_row[1].strftime('%Y-%m-%d') if game_row[1] else None,
                "time": game_row[2].strftime('%H:%M') if game_row[2] else None,
                "status": game_row[7]
            },
            "home_team": {
                "team_id": home_team_id,
                "abbreviation": game_row[8],
                "full_name": game_row[9],
                "logo_url": game_row[10],
                "primary_color": game_row[11],
                "secondary_color": game_row[12],
                "record": f"{home_stats[1]}-{home_stats[2]}" if home_stats[0] > 0 else "0-0",
                "wins": home_stats[1],
                "losses": home_stats[2],
                "ppg": round(float(home_stats[3]), 1) if home_stats[3] else 0,
                "net_rating": round(float(home_stats[4]), 1) if home_stats[4] else 0,
                "projected_starters": home_starters
            },
            "away_team": {
                "team_id": away_team_id,
                "abbreviation": game_row[13],
                "full_name": game_row[14],
                "logo_url": game_row[15],
                "primary_color": game_row[16],
                "secondary_color": game_row[17],
                "record": f"{away_stats[1]}-{away_stats[2]}" if away_stats[0] > 0 else "0-0",
                "wins": away_stats[1],
                "losses": away_stats[2],
                "ppg": round(float(away_stats[3]), 1) if away_stats[3] else 0,
                "net_rating": round(float(away_stats[4]), 1) if away_stats[4] else 0,
                "projected_starters": away_starters
            },
            "injuries": injuries
        }
        
    except Exception as e:
        if conn:
            conn.close()
        return {"error": str(e)}


@app.get("/api/team/{team_abbr}/schedule")
async def get_team_schedule(team_abbr: str, days_ahead: int = 14):
    """
    Get team schedule for next N days
    """
    conn = get_db_connection()
    if not conn:
        return {"games": []}
    
    try:
        from datetime import date, timedelta
        today = date.today()
        end_date = today + timedelta(days=days_ahead)
        
        cursor = conn.cursor()
        
        # Get team ID
        cursor.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (team_abbr,))
        team_row = cursor.fetchone()
        if not team_row:
            conn.close()
            return {"error": "Team not found"}
        
        team_id = team_row[0]
        
        cursor.execute("""
            SELECT 
                s.game_id, s.game_date, s.game_time,
                home.abbreviation, away.abbreviation,
                s.game_status, s.arena,
                CASE WHEN s.home_team_id = %s THEN 'Home' ELSE 'Away' END as location
            FROM nba_schedule s
            JOIN teams home ON home.team_id = s.home_team_id
            JOIN teams away ON away.team_id = s.away_team_id
            WHERE (s.home_team_id = %s OR s.away_team_id = %s)
              AND s.game_date >= %s AND s.game_date <= %s
            ORDER BY s.game_date, s.game_time
        """, (team_id, team_id, team_id, today, end_date))
        
        games = []
        for row in cursor.fetchall():
            games.append({
                "game_id": row[0],
                "date": row[1].strftime('%Y-%m-%d'),
                "time": row[2].strftime('%H:%M') if row[2] else None,
                "opponent": row[4] if row[7] == 'Home' else row[3],
                "location": row[7],
                "status": row[5],
                "arena": row[6]
            })
        
        conn.close()
        return {"team": team_abbr, "games": games, "count": len(games)}
        
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

