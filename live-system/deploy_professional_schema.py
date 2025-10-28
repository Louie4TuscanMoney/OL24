#!/usr/bin/env python3
"""
Professional PostgreSQL Schema Deployment Script
Ontologic XYZ - Data Science Company Standard

This script:
1. Deploys the professional V4 schema
2. Validates all tables and indexes
3. Runs performance optimizations
4. Reports deployment status

Usage:
    python deploy_professional_schema.py
"""

import os
import psycopg2
from datetime import datetime

# Database connection from environment
DATABASE_URL = os.environ.get('DATABASE_URL')

def deploy_schema():
    """Deploy the professional schema to PostgreSQL"""
    print("=" * 80)
    print("ONTOLOGIC XYZ - PROFESSIONAL SCHEMA DEPLOYMENT")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL environment variable not set!")
        print("Please set: export DATABASE_URL='your-postgres-url'")
        return False
    
    print(f"✅ Database URL found")
    print(f"📍 Target: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'localhost'}")
    print()
    
    try:
        # Connect to database
        print("🔌 Connecting to PostgreSQL...")
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False
        cur = conn.cursor()
        print("✅ Connected successfully")
        print()
        
        # Read schema file
        schema_file = 'database_schema_v4_PROFESSIONAL.sql'
        print(f"📖 Reading schema: {schema_file}")
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        print(f"✅ Schema loaded ({len(schema_sql)} characters)")
        print()
        
        # Execute schema
        print("🚀 Deploying schema...")
        print("   This may take 30-60 seconds...")
        cur.execute(schema_sql)
        conn.commit()
        print("✅ Schema deployed successfully!")
        print()
        
        # Validate deployment
        print("🔍 Validating deployment...")
        
        # Count tables
        cur.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        table_count = cur.fetchone()[0]
        print(f"   ✓ Tables created: {table_count}")
        
        # Count indexes
        cur.execute("""
            SELECT COUNT(*) 
            FROM pg_indexes 
            WHERE schemaname = 'public'
        """)
        index_count = cur.fetchone()[0]
        print(f"   ✓ Indexes created: {index_count}")
        
        # Count functions
        cur.execute("""
            SELECT COUNT(*) 
            FROM pg_proc p
            JOIN pg_namespace n ON p.pronamespace = n.oid
            WHERE n.nspname = 'public'
        """)
        function_count = cur.fetchone()[0]
        print(f"   ✓ Functions created: {function_count}")
        
        # Check ML tables
        print()
        print("🤖 ML System Tables:")
        ml_tables = ['ml_models', 'ml_predictions', 'prediction_performance']
        for table in ml_tables:
            cur.execute(f"""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = '{table}'
            """)
            exists = cur.fetchone()[0]
            status = "✅" if exists else "❌"
            print(f"   {status} {table}")
        
        # Check core tables
        print()
        print("📊 Core Tables:")
        core_tables = ['teams', 'players', 'games', 'player_box_scores', 
                       'player_season_stats', 'team_season_stats']
        for table in core_tables:
            cur.execute(f"""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = '{table}'
            """)
            exists = cur.fetchone()[0]
            status = "✅" if exists else "❌"
            print(f"   {status} {table}")
        
        # Run ANALYZE for performance
        print()
        print("⚡ Running performance optimization...")
        cur.execute("ANALYZE")
        conn.commit()
        print("✅ Database optimized")
        
        # Close connection
        cur.close()
        conn.close()
        
        print()
        print("=" * 80)
        print("✅ DEPLOYMENT SUCCESSFUL!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Run: python nba_nightly_pipeline.py (populate data)")
        print("  2. Set: MODEL_PATH environment variable (for ML)")
        print("  3. Deploy: git push origin main (Railway auto-deploy)")
        print()
        
        return True
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ DEPLOYMENT FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        if conn:
            conn.rollback()
            conn.close()
        return False


if __name__ == "__main__":
    success = deploy_schema()
    exit(0 if success else 1)

