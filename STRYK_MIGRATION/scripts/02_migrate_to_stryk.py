"""
STRYK MIGRATION STEP 2: MIGRATE TO POSTGRES
============================================

Migrates exported data to STRYK Postgres database

Usage:
    export POSTGRES_URL="postgresql://..."
    python 02_migrate_to_stryk.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text
from typing import Dict, List

# =====================================================
# CONFIGURATION
# =====================================================

INPUT_FILE = Path("../export/migration_data.json")
POSTGRES_URL = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")

if not POSTGRES_URL:
    print("❌ ERROR: POSTGRES_URL or DATABASE_URL environment variable not set!")
    print("   Set it with: export POSTGRES_URL='postgresql://...'")
    sys.exit(1)

# =====================================================
# MIGRATION FUNCTIONS
# =====================================================

def migrate_games(conn, games: List[Dict]) -> Dict[str, str]:
    """
    Migrate games to Postgres
    Returns mapping of external_id -> UUID
    """
    print(f"\n📦 Migrating {len(games)} games...")
    game_id_map = {}
    
    for i, g in enumerate(games, 1):
        try:
            game_sql = text("""
                INSERT INTO games (
                    external_id, home_team, away_team, home_score, away_score,
                    status, period, clock, spread, total, home_ml, away_ml,
                    spread_display, underdog_display,
                    home_implied_prob, away_implied_prob, home_no_vig_prob, away_no_vig_prob, vig_percentage,
                    source, is_q2_6min, can_predict, game_date
                ) VALUES (
                    :external_id, :home_team, :away_team, :home_score, :away_score,
                    :status, :period, :clock, :spread, :total, :home_ml, :away_ml,
                    :spread_display, :underdog_display,
                    :home_implied_prob, :away_implied_prob, :home_no_vig_prob, :away_no_vig_prob, :vig_percentage,
                    :source, :is_q2_6min, :can_predict, NOW()
                )
                ON CONFLICT (external_id, sport) DO UPDATE SET
                    home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score,
                    status = EXCLUDED.status,
                    period = EXCLUDED.period,
                    clock = EXCLUDED.clock,
                    spread = EXCLUDED.spread,
                    total = EXCLUDED.total,
                    home_ml = EXCLUDED.home_ml,
                    away_ml = EXCLUDED.away_ml,
                    spread_display = EXCLUDED.spread_display,
                    underdog_display = EXCLUDED.underdog_display,
                    home_implied_prob = EXCLUDED.home_implied_prob,
                    away_implied_prob = EXCLUDED.away_implied_prob,
                    home_no_vig_prob = EXCLUDED.home_no_vig_prob,
                    away_no_vig_prob = EXCLUDED.away_no_vig_prob,
                    vig_percentage = EXCLUDED.vig_percentage,
                    is_q2_6min = EXCLUDED.is_q2_6min,
                    can_predict = EXCLUDED.can_predict,
                    updated_at = NOW()
                RETURNING id
            """)
            
            result = conn.execute(game_sql, {
                "external_id": g.get("game_id", g.get("external_id", f"game_{i}")),
                "home_team": g.get("home_team"),
                "away_team": g.get("away_team"),
                "home_score": g.get("home_score"),
                "away_score": g.get("away_score"),
                "status": g.get("status", 2),
                "period": g.get("period"),
                "clock": g.get("clock"),
                "spread": g.get("spread"),
                "total": g.get("total"),
                "home_ml": g.get("home_ml") or g.get("moneyline_home"),
                "away_ml": g.get("away_ml") or g.get("moneyline_away"),
                "spread_display": g.get("spread_display"),
                "underdog_display": g.get("underdog_display"),
                "home_implied_prob": g.get("home_implied_prob"),
                "away_implied_prob": g.get("away_implied_prob"),
                "home_no_vig_prob": g.get("home_no_vig_prob"),
                "away_no_vig_prob": g.get("away_no_vig_prob"),
                "vig_percentage": g.get("vig_percentage"),
                "source": g.get("source", "BetOnline"),
                "is_q2_6min": g.get("is_q2_6min", False),
                "can_predict": g.get("can_predict", False)
            })
            
            game_id = result.fetchone()[0]
            game_id_map[g.get("game_id", g.get("external_id"))] = game_id
            
            if i % 10 == 0:
                print(f"   ✅ {i}/{len(games)} games migrated...")
        
        except Exception as e:
            print(f"   ⚠️  Error migrating game {i}: {e}")
            continue
    
    print(f"✅ Migrated {len(game_id_map)} games successfully")
    return game_id_map

def migrate_predictions(conn, opportunities: List[Dict], game_id_map: Dict[str, str]):
    """Migrate ML predictions"""
    if not opportunities:
        print("\n⚠️  No opportunities to migrate")
        return
    
    print(f"\n📦 Migrating {len(opportunities)} predictions...")
    
    for i, opp in enumerate(opportunities, 1):
        try:
            external_id = opp.get("game_id")
            game_id = game_id_map.get(external_id)
            
            if not game_id:
                print(f"   ⚠️  No game found for prediction {i} (game_id: {external_id})")
                continue
            
            pred_sql = text("""
                INSERT INTO predictions (
                    game_id, model_version, spread_pred, mamba_score,
                    mamba_probability, market_probability, kelly_edge, kelly_bet_size,
                    recommended_bet, bet_confidence, archetype, risk_score,
                    is_q2_6min, period, clock, features
                ) VALUES (
                    :game_id, :model_version, :spread_pred, :mamba_score,
                    :mamba_probability, :market_probability, :kelly_edge, :kelly_bet_size,
                    :recommended_bet, :bet_confidence, :archetype, :risk_score,
                    :is_q2_6min, :period, :clock, :features::jsonb
                )
            """)
            
            conn.execute(pred_sql, {
                "game_id": game_id,
                "model_version": opp.get("model_version", "mamba_v1"),
                "spread_pred": opp.get("prediction") or opp.get("mamba_prediction"),
                "mamba_score": opp.get("mamba_score"),
                "mamba_probability": opp.get("mamba_probability") or opp.get("calibrated_prob"),
                "market_probability": opp.get("market_probability") or opp.get("implied_prob"),
                "kelly_edge": opp.get("kelly_edge") or opp.get("edge"),
                "kelly_bet_size": opp.get("kelly_bet_size") or opp.get("stake"),
                "recommended_bet": opp.get("recommended_bet") or opp.get("bet"),
                "bet_confidence": opp.get("bet_confidence") or opp.get("confidence"),
                "archetype": opp.get("archetype") or opp.get("game_archetype"),
                "risk_score": opp.get("risk_score"),
                "is_q2_6min": opp.get("is_q2_6min", True),
                "period": opp.get("period"),
                "clock": opp.get("clock"),
                "features": json.dumps(opp.get("features", []))
            })
            
            if i % 10 == 0:
                print(f"   ✅ {i}/{len(opportunities)} predictions migrated...")
        
        except Exception as e:
            print(f"   ⚠️  Error migrating prediction {i}: {e}")
            continue
    
    print(f"✅ Migrated predictions successfully")

def migrate_mamba_scores(conn, mamba_scores: List[Dict], game_id_map: Dict[str, str]):
    """Migrate stored Mamba scores"""
    if not mamba_scores:
        print("\n⚠️  No Mamba scores to migrate")
        return
    
    print(f"\n📦 Migrating {len(mamba_scores)} Mamba scores...")
    
    for i, score in enumerate(mamba_scores, 1):
        try:
            external_id = score.get("game_id") or score.get("external_id")
            game_id = game_id_map.get(external_id)
            
            if not game_id:
                print(f"   ⚠️  No game found for Mamba score {i} (game_id: {external_id})")
                continue
            
            score_sql = text("""
                INSERT INTO mamba_scores (
                    game_id, external_id, home_team, away_team,
                    mamba_score, period, clock
                ) VALUES (
                    :game_id, :external_id, :home_team, :away_team,
                    :mamba_score, :period, :clock
                )
                ON CONFLICT (external_id, period, clock) DO NOTHING
            """)
            
            conn.execute(score_sql, {
                "game_id": game_id,
                "external_id": external_id,
                "home_team": score.get("home_team"),
                "away_team": score.get("away_team"),
                "mamba_score": score.get("mamba_score") or score.get("score"),
                "period": score.get("period"),
                "clock": score.get("clock")
            })
            
            if i % 10 == 0:
                print(f"   ✅ {i}/{len(mamba_scores)} Mamba scores migrated...")
        
        except Exception as e:
            print(f"   ⚠️  Error migrating Mamba score {i}: {e}")
            continue
    
    print(f"✅ Migrated Mamba scores successfully")

# =====================================================
# MAIN MIGRATION
# =====================================================

def run_migration():
    """Run complete migration"""
    print("=" * 60)
    print("🚀 STRYK MIGRATION: MIGRATING TO POSTGRES")
    print("=" * 60)
    
    # Load exported data
    print(f"\n📂 Loading {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        print(f"❌ ERROR: {INPUT_FILE} not found!")
        print("   Run 01_export_old_data.py first")
        sys.exit(1)
    
    with open(INPUT_FILE) as f:
        export_data = json.load(f)
    
    data = export_data.get("data", {})
    print(f"✅ Loaded export from {export_data.get('export_timestamp')}")
    
    # Connect to Postgres
    print(f"\n🔌 Connecting to Postgres...")
    engine = create_engine(POSTGRES_URL)
    
    # Run migration in transaction
    with engine.begin() as conn:
        print("✅ Connected to Postgres")
        
        # 1. Migrate games
        games = data.get("live_games", [])
        if isinstance(games, dict):
            games = [games]
        game_id_map = migrate_games(conn, games)
        
        # 2. Migrate predictions
        opportunities = data.get("opportunities", [])
        if isinstance(opportunities, dict):
            opportunities = [opportunities]
        migrate_predictions(conn, opportunities, game_id_map)
        
        # 3. Migrate Mamba scores
        mamba_scores = data.get("mamba_scores", [])
        if isinstance(mamba_scores, dict):
            mamba_scores = [mamba_scores]
        migrate_mamba_scores(conn, mamba_scores, game_id_map)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ MIGRATION COMPLETE")
    print("=" * 60)
    print(f"📊 Games: {len(game_id_map)}")
    print(f"📊 Predictions: {len(opportunities)}")
    print(f"📊 Mamba scores: {len(mamba_scores)}")
    print("\n🎯 Next step: Run 03_warmup_redis.py")
    print("=" * 60)

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    try:
        run_migration()
    except KeyboardInterrupt:
        print("\n\n⚠️  Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

