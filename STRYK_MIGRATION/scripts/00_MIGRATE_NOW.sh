#!/bin/bash

# =====================================================
# STRYK MIGRATION: ONE-COMMAND EXECUTION
# =====================================================
# Run all migration steps in sequence
# Usage: bash 00_MIGRATE_NOW.sh
# =====================================================

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STRYK MIGRATION: COMPLETE AUTOMATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found. Please install Python 3.8+."
    exit 1
fi

if ! command -v railway &> /dev/null; then
    echo "❌ railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

echo "✅ Prerequisites OK"
echo ""

# Check environment variables
if [ -z "$POSTGRES_URL" ]; then
    echo "⚠️  POSTGRES_URL not set. Please run:"
    echo "   export POSTGRES_URL='postgresql://...'"
    echo ""
    read -p "Enter POSTGRES_URL now: " POSTGRES_URL
    export POSTGRES_URL
fi

if [ -z "$REDIS_URL" ]; then
    echo "⚠️  REDIS_URL not set. Please run:"
    echo "   export REDIS_URL='redis://...'"
    echo ""
    read -p "Enter REDIS_URL now: " REDIS_URL
    export REDIS_URL
fi

echo "✅ Environment variables OK"
echo ""

# Confirm migration
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  WARNING: This will migrate your backend to STRYK."
echo "   Make sure you have:"
echo "   - Backed up your current data"
echo "   - Tested the Postgres schema"
echo "   - Your old backend is still running (for rollback)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
read -p "Continue with migration? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Migration cancelled."
    exit 0
fi

echo ""
echo "🎬 Starting migration in 3 seconds..."
sleep 3

# =====================================================
# STEP 1: EXPORT OLD DATA
# =====================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 STEP 1: EXPORTING OLD DATA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 01_export_old_data.py

if [ $? -ne 0 ]; then
    echo "❌ Export failed. Check your backend URL in 01_export_old_data.py"
    exit 1
fi

echo "✅ Export complete"

# =====================================================
# STEP 2: MIGRATE TO POSTGRES
# =====================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🗄️  STEP 2: MIGRATING TO POSTGRES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 02_migrate_to_stryk.py

if [ $? -ne 0 ]; then
    echo "❌ Migration failed. Check Postgres connection."
    exit 1
fi

echo "✅ Postgres migration complete"

# =====================================================
# STEP 3: WARM REDIS CACHE
# =====================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔥 STEP 3: WARMING REDIS CACHE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 03_warmup_redis.py

if [ $? -ne 0 ]; then
    echo "❌ Redis warmup failed. Check Redis connection."
    exit 1
fi

echo "✅ Redis warmup complete"

# =====================================================
# SUMMARY
# =====================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 MIGRATION COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Migration Summary:"
echo "   ✅ Data exported from old backend"
echo "   ✅ Migrated to Postgres"
echo "   ✅ Redis cache warmed"
echo ""
echo "🎯 Next Steps:"
echo "   1. Update your backend code to use Postgres + Redis"
echo "   2. Deploy backend: railway up"
echo "   3. Deploy frontend: vercel --prod"
echo "   4. Verify everything works"
echo ""
echo "📁 Export file saved at: ../export/migration_data.json"
echo "   Keep this file for 30 days (for rollback if needed)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STRYK IS READY TO GO LIVE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

