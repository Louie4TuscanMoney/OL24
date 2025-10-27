"""
ONTORISK API

Purpose: REST API for live predictions and betting recommendations
Author: Ontologic XYZ
Date: October 20, 2025

Usage:
    python ontorisk_api.py
    
Then access:
    http://localhost:8000/docs for API documentation
    http://localhost:8000/predict for predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import uvicorn

from ontorisk_complete_system import OntoRiskCompleteSystem


# Initialize FastAPI
app = FastAPI(
    title="OntoRisk API",
    description="NBA Betting Risk Management & Prediction API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OntoRisk system (global)
system = None


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: List[float]
    spread_line: float
    home_team: str
    away_team: str


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: float
    spread_line: float
    edge: float
    p_win: float
    kelly_edge: float
    confidence_interval: List[float]
    bet_recommended: bool
    bet_side: str
    bet_line: str
    recommended_stake: float
    home_team: str
    away_team: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    model_loaded: bool
    mae: float


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global system
    
    print("\n🚀 Starting OntoRisk API...")
    
    try:
        system = OntoRiskCompleteSystem(
            model_path="../Action/HYBRID_ULTIMATE_V2_CLEAN.pkl",
            data_path="../Action/ULTRA_ENHANCED_PATTERNS_V3_CHRONOLOGICAL.pkl",
            mae=9.029,
            starting_bankroll=10000,
            kelly_fraction=0.25,
            min_edge=5.0,
            min_p_win=0.55
        )
        print("✅ OntoRisk system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        system = None


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "version": "1.0.0",
        "model_loaded": system is not None and system.model is not None,
        "mae": system.mae if system is not None else 0
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction for a game
    
    Example request:
    ```json
    {
        "features": [0.5, 1.2, -0.3, ...],  // 18 features
        "spread_line": -3.5,
        "home_team": "LAL",
        "away_team": "BOS"
    }
    ```
    """
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if system.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert features to numpy array
    features = np.array(request.features)
    
    # Make prediction
    try:
        result = system.predict_live_game(
            game_features=features,
            spread_line=request.spread_line,
            home_team=request.home_team,
            away_team=request.away_team
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/backtest/summary")
async def backtest_summary():
    """
    Get summary of historical backtest results
    """
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Run quick backtest
    try:
        results = system.run_backtest(use_synthetic_spreads=True)
        
        if results is None:
            return {"error": "No backtest results available"}
        
        return {
            "total_games": results.total_games,
            "games_bet": results.games_bet,
            "win_rate": results.win_rate,
            "roi": results.roi,
            "total_profit": results.total_profit,
            "sharpe_ratio": results.sharpe_ratio,
            "max_drawdown": results.max_drawdown
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@app.get("/config")
async def get_config():
    """Get current system configuration"""
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "mae": system.mae,
        "starting_bankroll": system.starting_bankroll,
        "kelly_fraction": system.kelly_fraction,
        "min_edge": system.min_edge,
        "min_p_win": system.min_p_win
    }


@app.post("/config/update")
async def update_config(
    min_edge: Optional[float] = None,
    min_p_win: Optional[float] = None,
    kelly_fraction: Optional[float] = None
):
    """Update system configuration"""
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if min_edge is not None:
        system.min_edge = min_edge
    
    if min_p_win is not None:
        system.min_p_win = min_p_win
    
    if kelly_fraction is not None:
        system.kelly_fraction = kelly_fraction
        system.backtest_engine.kelly_fraction = kelly_fraction
    
    return {
        "message": "Configuration updated",
        "new_config": {
            "min_edge": system.min_edge,
            "min_p_win": system.min_p_win,
            "kelly_fraction": system.kelly_fraction
        }
    }


def start_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server"""
    print("\n" + "="*80)
    print("🔥 STARTING ONTORISK API")
    print("="*80)
    print(f"\nAPI will be available at:")
    print(f"  • http://localhost:{port}/")
    print(f"  • http://localhost:{port}/docs (Swagger UI)")
    print(f"  • http://localhost:{port}/redoc (ReDoc)")
    print("\n" + "="*80)
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_api()

