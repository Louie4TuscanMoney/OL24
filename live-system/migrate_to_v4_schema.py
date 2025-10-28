#!/usr/bin/env python3
"""
Safe Migration from V3 to V4 Schema
Drops conflicting tables and recreates with new structure
"""

import os
import psycopg2

DATABASE_URL = os.environ.get('DATABASE_URL')

def migrate_schema():
    print("=" * 80)
    print("ONTOLOGIC XYZ - SCHEMA MIGRATION V3 → V4")
    print("=" * 80)
    print()
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set!")
        return False
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True  # ✅ AUTO-COMMIT: Each statement is its own transaction
        cur = conn.cursor()
        
        print("🗑️  Dropping old schema (if exists)...")
        print()
        
        # Drop materialized views first
        print("   📊 Dropping materialized views...")
        try:
            cur.execute("DROP MATERIALIZED VIEW IF EXISTS player_last10 CASCADE")
            print("      ✓ Dropped player_last10")
        except Exception as e:
            print(f"      ⚠️  Could not drop player_last10: {e}")
        
        print()
        print("   📦 Dropping tables...")
        
        # Drop tables in reverse dependency order
        tables_to_drop = [
            'api_requests',
            'pipeline_runs',
            'prediction_performance',
            'ml_predictions',
            'ml_models',
            'nba_schedule',
            'team_depth_charts',
            'player_injuries',
            'team_season_stats',
            'player_season_stats',
            'player_box_scores_2026_27',
            'player_box_scores_2025_26',
            'player_box_scores_2024_25',
            'player_box_scores',
            'games',
            'seasons',
            'players',
            'teams'
        ]
        
        for table in tables_to_drop:
            try:
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                print(f"      ✓ Dropped {table}")
            except Exception as e:
                print(f"      ⚠️  Could not drop {table}: {e}")
        
        print()
        print("✅ Old schema cleaned!")
        print()
        
        print("📖 Reading V4 schema...")
        with open('database_schema_v4_PROFESSIONAL.sql', 'r') as f:
            schema_sql = f.read()
        print(f"✅ Schema loaded ({len(schema_sql)} characters)")
        print()
        
        print("🚀 Deploying V4 schema...")
        print("   This may take 30-60 seconds...")
        print()
        
        # Execute schema (will succeed because tables are dropped)
        conn.autocommit = False  # Switch to transaction mode for schema deployment
        cur.execute(schema_sql)
        conn.commit()
        
        print("✅ V4 Schema deployed!")
        print()
        
        # Verify
        conn.autocommit = True
        print("🔍 Verifying deployment...")
        cur.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        table_count = cur.fetchone()[0]
        print(f"   ✓ Tables created: {table_count}")
        
        cur.execute("""
            SELECT COUNT(*) 
            FROM pg_indexes 
            WHERE schemaname = 'public'
        """)
        index_count = cur.fetchone()[0]
        print(f"   ✓ Indexes created: {index_count}")
        
        # Check ML tables
        print()
        print("   🤖 ML System Tables:")
        ml_tables = ['ml_models', 'ml_predictions', 'prediction_performance']
        for table in ml_tables:
            cur.execute(f"""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = '{table}'
            """)
            exists = cur.fetchone()[0]
            status = "✅" if exists else "❌"
            print(f"      {status} {table}")
        
        print()
        print("=" * 80)
        print("✅ MIGRATION SUCCESSFUL!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Populate data: python3 nba_nightly_pipeline.py")
        print("  2. Check API: curl https://ol24-production.up.railway.app/api/stats/teams")
        print()
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ MIGRATION FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        if conn:
            conn.rollback()
            conn.close()
        return False


if __name__ == "__main__":
    success = migrate_schema()
    exit(0 if success else 1)
