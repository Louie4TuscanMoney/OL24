"""
Deploy schema to PostgreSQL (no psql required!)
"""

import os
import psycopg2

DATABASE_URL = os.environ.get('DATABASE_URL')

if not DATABASE_URL:
    print("❌ DATABASE_URL not set!")
    print("\nRun this first:")
    print('export DATABASE_URL="postgresql://postgres:lubdUxBtqzIWOynpPZiEXqFWPJvBlkVb@postgres.railway.internal:5432/railway"')
    exit(1)

print("\n" + "="*80)
print("📊 DEPLOYING SCHEMA TO POSTGRESQL")
print("="*80 + "\n")

# Read schema file
with open('database_schema_v3_SELF_COMPUTED.sql', 'r') as f:
    schema_sql = f.read()

# Connect and execute
print("Connecting to database...")
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

print("✅ Connected!")
print("\nExecuting schema (this may take 30 seconds)...\n")

# Execute the entire schema
cursor.execute(schema_sql)
conn.commit()

print("\n" + "="*80)
print("✅ SCHEMA DEPLOYED SUCCESSFULLY!")
print("="*80)

# Verify tables were created
cursor.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    ORDER BY table_name
""")

tables = cursor.fetchall()
print(f"\n📊 Created {len(tables)} tables:")
for table in tables:
    print(f"   ✅ {table[0]}")

# Check teams were inserted
cursor.execute("SELECT COUNT(*) FROM teams")
team_count = cursor.fetchone()[0]
print(f"\n📊 Inserted {team_count} teams")

cursor.close()
conn.close()

print("\n" + "="*80)
print("🎉 DATABASE READY!")
print("="*80 + "\n")

