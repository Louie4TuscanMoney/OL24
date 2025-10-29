"""
Safe schema deployment with migration support
Handles existing tables and adds missing columns
"""

import os
import psycopg2

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    print("❌ DATABASE_URL not set!")
    exit(1)

print("🚀 Deploying schema to Railway PostgreSQL (migration-safe)...")
print(f"   Connection: yamabiko.proxy.rlwy.net:37192")
print()

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

try:
    # Step 1: Drop ALL existing tables for clean deployment
    print("   🔧 Dropping all existing tables for clean deployment...")
    
    # Get all existing tables
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    """)
    existing_tables = [t[0] for t in cur.fetchall()]
    
    if existing_tables:
        print(f"      • Found {len(existing_tables)} existing tables")
        # Drop all tables
        for table in existing_tables:
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            print(f"      • Dropped {table}")
    else:
        print("      • No existing tables found")
    
    conn.commit()
    print("   ✅ Cleanup complete")
    print()
    
    # Step 2: Deploy full schema
    print("   ⏳ Creating tables from schema...")
    
    with open('database_schema_OPTIMIZED_FOR_FRONTEND.sql', 'r') as f:
        schema_sql = f.read()
    
    cur.execute(schema_sql)
    conn.commit()
    
    print("   ✅ Schema deployed successfully!")
    print()
    
    # Step 3: Verify
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name")
    tables = [t[0] for t in cur.fetchall()]
    print(f"   📊 Created {len(tables)} tables:")
    for t in tables:
        print(f"      • {t}")
    
    print()
    
    # Verify teams
    cur.execute("SELECT COUNT(*) FROM teams")
    teams_count = cur.fetchone()[0]
    print(f"   ✅ Teams: {teams_count}/30")
    
    # Check logos
    cur.execute("SELECT COUNT(*) FROM teams WHERE logo_url IS NOT NULL AND primary_color IS NOT NULL")
    teams_with_visuals = cur.fetchone()[0]
    print(f"   ✅ Teams with logos & colors: {teams_with_visuals}/30")
    
    # Verify critical columns exist
    cur.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'player_injuries' AND column_name = 'is_active'
    """)
    if cur.fetchone():
        print(f"   ✅ player_injuries.is_active exists")
    
    cur.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'player_last10' AND column_name = 'game_rank'
    """)
    if cur.fetchone():
        print(f"   ✅ player_last10.game_rank exists")
    
    # Show sample
    cur.execute("SELECT abbreviation, full_name, primary_color FROM teams WHERE abbreviation = 'LAL'")
    lal = cur.fetchone()
    if lal:
        print()
        print(f"   🏀 Sample: {lal[0]} - {lal[1]} ({lal[2]})")
    
    print()
    print("="*80)
    print("✅ SCHEMA DEPLOYED SUCCESSFULLY!")
    print("="*80)
    print()
    print("🚀 Next: Populate the database")
    print("   Run: python3 populate_database_for_frontend.py")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    conn.rollback()

finally:
    cur.close()
    conn.close()

