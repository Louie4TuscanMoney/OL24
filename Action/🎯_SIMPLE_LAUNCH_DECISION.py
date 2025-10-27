#!/usr/bin/env python3
"""
🎯 SIMPLE LAUNCH DECISION
Test existing Dejavu model + new XGBoost on 2025 data
Make GO/CAUTIOUS/NO-GO decision
"""

import sys
sys.path.insert(0, '1. ML/1. Dejavu Deployment')

import pickle
import numpy as np
from dejavu_model import DejavuForecaster
import xgboost as xgb

print("="*70)
print("🎯 LAUNCH DECISION - Testing on 2025 Data")
print("="*70)
print()

# Load models
print("[1/4] Loading models...")
try:
    dejavu = DejavuForecaster.load('1. ML/1. Dejavu Deployment/dejavu_FINAL_k500.pkl')
    print("✅ Loaded Dejavu (FINAL prediction model)")
except:
    print("❌ Dejavu load failed")
    dejavu = None

try:
    xgboost_model = xgb.XGBRegressor()
    xgboost_model.load_model('xgboost_simple_v1.json')
    print("✅ Loaded XGBoost")
except:
    print("⚠️  XGBoost load failed - using Dejavu only")
    xgboost_model = None

print()

# Load 2025 data (preseason games from our extracted data)
print("[2/4] Loading 2025 preseason data...")
with open('ULTRA_OPTIMIZED_PATTERNS_2021_2025.pkl', 'rb') as f:
    all_patterns = pickle.load(f)

# Filter for 2025 preseason (game IDs starting with 0042400 or similar)
test_games = [p for p in all_patterns if str(p.get('season', '')).startswith('2024')]

if len(test_games) == 0:
    print("⚠️  No 2025 games found, using most recent 100 games as test")
    test_games = all_patterns[-100:]

print(f"✅ Found {len(test_games)} test games")
print()

# Test Dejavu
print("[3/4] Testing Dejavu model...")
dejavu_errors = []

for game in test_games:
    pattern = game.get('pattern', [])
    actual = game.get('diff_at_final')
    
    if len(pattern) != 18 or actual is None:
        continue
    
    try:
        pred = dejavu.predict(pattern)
        error = abs(pred - actual)
        dejavu_errors.append(error)
    except:
        continue

if dejavu_errors:
    dejavu_mae = np.mean(dejavu_errors)
    print(f"📊 Dejavu MAE: {dejavu_mae:.2f} points")
    print(f"   Sample size: {len(dejavu_errors)} games")
else:
    dejavu_mae = 999
    print("❌ Dejavu testing failed")

print()

# Test XGBoost (if available)
print("[4/4] Testing XGBoost model...")
if xgboost_model:
    xgboost_errors = []
    
    for game in test_games:
        pattern = game.get('pattern', [])
        actual = game.get('diff_at_final')
        stat = game.get('pattern_statistical', {})
        qual = game.get('quality_metrics', {})
        team = game.get('team_features', {})
        player = game.get('player_features', {})
        
        if len(pattern) != 18 or actual is None:
            continue
        
        try:
            # Build feature vector (same as training)
            features = []
            features.extend(pattern)
            features.extend([
                stat.get('mean', 0), stat.get('std', 0), 
                stat.get('trend', 0), stat.get('volatility', 0)
            ])
            features.append(1.0 if qual.get('quality_grade') == 'A' else 0.5)
            features.extend([
                team.get('home_off_rating', 110.0), team.get('home_def_rating', 110.0),
                team.get('away_off_rating', 110.0), team.get('away_def_rating', 110.0),
                team.get('home_win_pct', 0.5), team.get('away_win_pct', 0.5)
            ])
            features.extend([
                player.get('home_star_count', 0), player.get('away_star_count', 0),
                player.get('home_avg_tier', 3.0), player.get('away_avg_tier', 3.0),
                player.get('home_depth', 0), player.get('away_depth', 0)
            ])
            
            pred = xgboost_model.predict(np.array([features]))[0]
            error = abs(pred - actual)
            xgboost_errors.append(error)
        except Exception as e:
            continue
    
    if xgboost_errors:
        xgboost_mae = np.mean(xgboost_errors)
        print(f"📊 XGBoost MAE: {xgboost_mae:.2f} points")
        print(f"   Sample size: {len(xgboost_errors)} games")
    else:
        xgboost_mae = 999
        print("❌ XGBoost testing failed")
else:
    xgboost_mae = 999
    print("⏭️  XGBoost not available")

print()
print("="*70)
print("🎯 LAUNCH DECISION")
print("="*70)
print()

# Determine best model
best_mae = min(dejavu_mae, xgboost_mae)
best_model = "Dejavu" if dejavu_mae < xgboost_mae else "XGBoost"

print(f"📊 Best Model: {best_model}")
print(f"📊 Best MAE: {best_mae:.2f} points")
print()

# Make decision
if best_mae < 7.0:
    decision = "🟢 GO FOR LAUNCH"
    risk_mode = "AGGRESSIVE"
    max_bet = 200
    portfolio_cap = 2000
    kelly = 0.25
    recommendation = "Launch Monday with confidence. MAE is excellent (<7)."
elif best_mae < 9.0:
    decision = "🟡 CAUTIOUS LAUNCH"
    risk_mode = "CONSERVATIVE"
    max_bet = 50
    portfolio_cap = 300
    kelly = 0.10
    recommendation = "Launch Monday conservatively. MAE is acceptable (7-9)."
else:
    decision = "🔴 NO-GO / MICRO-TEST"
    risk_mode = "ULTRA-CONSERVATIVE"
    max_bet = 10
    portfolio_cap = 50
    kelly = 0.05
    recommendation = "Consider paper trading or micro-bets only. MAE is high (>9)."

print(f"🎯 DECISION: {decision}")
print(f"📊 MAE: {best_mae:.2f}")
print(f"🛡️  Risk Mode: {risk_mode}")
print()
print("Risk Configuration:")
print(f"  Max bet per game: ${max_bet}")
print(f"  Portfolio cap per day: ${portfolio_cap}")
print(f"  Kelly fraction: {kelly}")
print()
print(f"Recommendation: {recommendation}")
print()

# Save decision
decision_data = {
    'decision': decision,
    'mae': best_mae,
    'best_model': best_model,
    'dejavu_mae': dejavu_mae,
    'xgboost_mae': xgboost_mae,
    'risk_mode': risk_mode,
    'max_bet': max_bet,
    'portfolio_cap': portfolio_cap,
    'kelly_fraction': kelly,
    'recommendation': recommendation,
    'n_games': len(test_games)
}

with open('LAUNCH_DECISION.pkl', 'wb') as f:
    pickle.dump(decision_data, f)

print("✅ Decision saved to LAUNCH_DECISION.pkl")
print()

# Create risk configuration file
risk_config = f"""# Auto-generated risk configuration
# Based on {best_model} model with MAE {best_mae:.2f}

RISK_MODE = "{risk_mode}"
MAX_BET_PER_GAME = {max_bet}
PORTFOLIO_CAP_PER_DAY = {portfolio_cap}
KELLY_FRACTION = {kelly}
CONFIDENCE_THRESHOLD = {0.7 if best_mae < 7 else 0.85 if best_mae < 9 else 0.95}

# Model selection
USE_MODEL = "{best_model.lower()}"
DEJAVU_MAE = {dejavu_mae:.2f}
XGBOOST_MAE = {xgboost_mae:.2f}
"""

with open('risk_configuration.py', 'w') as f:
    f.write(risk_config)

print("✅ Risk configuration saved to risk_configuration.py")
print()
print("="*70)
print("✅ ANALYSIS COMPLETE")
print("="*70)

