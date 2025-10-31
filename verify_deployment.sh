#!/bin/bash

# =============================================================================
# COMPLETE DEPLOYMENT VERIFICATION SCRIPT
# Run this after deploying to Railway to verify everything works
# =============================================================================

set -e

echo "================================================================================"
echo "🔍 RAILWAY POSTGRESQL DEPLOYMENT VERIFICATION"
echo "================================================================================"
echo ""

# Check DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "❌ DATABASE_URL not set!"
    echo "   Set it with: export DATABASE_URL='postgresql://...'"
    exit 1
fi

echo "✅ DATABASE_URL is set"
echo ""

# =============================================================================
# DATABASE CHECKS
# =============================================================================

echo "📊 DATABASE CHECKS"
echo "================================================================================\n"

# Check 1: Teams count
echo "1️⃣  Checking teams table..."
TEAMS_COUNT=$(psql $DATABASE_URL -t -c "SELECT COUNT(*) FROM teams;")
if [ "$TEAMS_COUNT" -eq 30 ]; then
    echo "   ✅ Teams: $TEAMS_COUNT/30"
else
    echo "   ❌ Teams: $TEAMS_COUNT/30 (expected 30)"
fi

# Check 2: Teams with visuals
TEAMS_WITH_VISUALS=$(psql $DATABASE_URL -t -c "SELECT COUNT(*) FROM teams WHERE logo_url IS NOT NULL AND primary_color IS NOT NULL;")
if [ "$TEAMS_WITH_VISUALS" -eq 30 ]; then
    echo "   ✅ Teams with logos & colors: $TEAMS_WITH_VISUALS/30"
else
    echo "   ⚠️  Teams with logos & colors: $TEAMS_WITH_VISUALS/30"
fi

# Check 3: Players count
echo "2️⃣  Checking players table..."
PLAYERS_COUNT=$(psql $DATABASE_URL -t -c "SELECT COUNT(*) FROM players WHERE is_active = TRUE;")
if [ "$PLAYERS_COUNT" -gt 400 ]; then
    echo "   ✅ Active players: $PLAYERS_COUNT"
else
    echo "   ⚠️  Active players: $PLAYERS_COUNT (expected 400+)"
fi

# Check 4: Player stats
STATS_COUNT=$(psql $DATABASE_URL -t -c "SELECT COUNT(*) FROM player_season_stats WHERE season_id = '2025-26';")
if [ "$STATS_COUNT" -gt 400 ]; then
    echo "   ✅ Players with stats: $STATS_COUNT"
else
    echo "   ⚠️  Players with stats: $STATS_COUNT (expected 400+)"
fi

# Check 5: Schedule
echo "3️⃣  Checking schedule table..."
SCHEDULE_COUNT=$(psql $DATABASE_URL -t -c "SELECT COUNT(*) FROM nba_schedule;")
if [ "$SCHEDULE_COUNT" -gt 0 ]; then
    echo "   ✅ Scheduled games: $SCHEDULE_COUNT"
else
    echo "   ⚠️  Scheduled games: $SCHEDULE_COUNT (run population script)"
fi

# Check 6: Injuries
echo "4️⃣  Checking injuries table..."
INJURIES_COUNT=$(psql $DATABASE_URL -t -c "SELECT COUNT(*) FROM player_injuries WHERE is_active = TRUE;")
echo "   ✅ Active injuries: $INJURIES_COUNT"

echo ""

# =============================================================================
# API ENDPOINT CHECKS
# =============================================================================

echo "🌐 API ENDPOINT CHECKS"
echo "================================================================================"
echo ""

API_BASE="https://ol24-production.up.railway.app"

# Check 1: Health
echo "1️⃣  Testing health endpoint..."
HEALTH=$(curl -s "${API_BASE}/" | jq -r '.status // "error"')
if [ "$HEALTH" = "online" ]; then
    echo "   ✅ Health: $HEALTH"
else
    echo "   ❌ Health check failed"
fi

# Check 2: Teams endpoint
echo "2️⃣  Testing /api/stats/teams..."
TEAMS_API=$(curl -s "${API_BASE}/api/stats/teams" | jq -r '.count // 0')
if [ "$TEAMS_API" -eq 30 ]; then
    echo "   ✅ Teams API: $TEAMS_API teams"
else
    echo "   ⚠️  Teams API: $TEAMS_API teams (expected 30)"
fi

# Check 3: Injuries endpoint
echo "3️⃣  Testing /api/injuries..."
INJURIES_API=$(curl -s "${API_BASE}/api/injuries" | jq -r '.count // 0')
echo "   ✅ Injuries API: $INJURIES_API injuries"

# Check 4: Schedule endpoint
echo "4️⃣  Testing /api/schedule..."
SCHEDULE_API=$(curl -s "${API_BASE}/api/schedule" | jq -r '.count // 0')
if [ "$SCHEDULE_API" -gt 0 ]; then
    echo "   ✅ Schedule API: $SCHEDULE_API games"
else
    echo "   ⚠️  Schedule API: $SCHEDULE_API games"
fi

echo ""

# =============================================================================
# PERFORMANCE CHECKS
# =============================================================================

echo "⚡ PERFORMANCE CHECKS"
echo "================================================================================"
echo ""

echo "Testing response times..."

# Teams endpoint
TEAMS_TIME=$(curl -s -o /dev/null -w "%{time_total}" "${API_BASE}/api/stats/teams")
echo "   /api/stats/teams: ${TEAMS_TIME}s"

# Injuries endpoint
INJURIES_TIME=$(curl -s -o /dev/null -w "%{time_total}" "${API_BASE}/api/injuries")
echo "   /api/injuries: ${INJURIES_TIME}s"

# Schedule endpoint
SCHEDULE_TIME=$(curl -s -o /dev/null -w "%{time_total}" "${API_BASE}/api/schedule")
echo "   /api/schedule: ${SCHEDULE_TIME}s"

echo ""

# =============================================================================
# SUMMARY
# =============================================================================

echo "================================================================================"
echo "📋 DEPLOYMENT CHECKLIST"
echo "================================================================================"
echo ""

# Calculate checks
TOTAL_CHECKS=8
PASSED_CHECKS=0

# Check results
if [ "$TEAMS_COUNT" -eq 30 ]; then ((PASSED_CHECKS++)); fi
if [ "$TEAMS_WITH_VISUALS" -eq 30 ]; then ((PASSED_CHECKS++)); fi
if [ "$PLAYERS_COUNT" -gt 400 ]; then ((PASSED_CHECKS++)); fi
if [ "$STATS_COUNT" -gt 400 ]; then ((PASSED_CHECKS++)); fi
if [ "$SCHEDULE_COUNT" -gt 0 ]; then ((PASSED_CHECKS++)); fi
if [ "$HEALTH" = "online" ]; then ((PASSED_CHECKS++)); fi
if [ "$TEAMS_API" -eq 30 ]; then ((PASSED_CHECKS++)); fi
if [ "$SCHEDULE_API" -gt 0 ]; then ((PASSED_CHECKS++)); fi

echo "✅ Passed: $PASSED_CHECKS/$TOTAL_CHECKS checks"
echo ""

if [ $PASSED_CHECKS -eq $TOTAL_CHECKS ]; then
    echo "🎉 ALL CHECKS PASSED! Database is fully operational!"
    echo ""
    echo "Next steps:"
    echo "  1. Open frontend: https://ontologicxyz.com"
    echo "  2. Check /stats page for team cards with logos"
    echo "  3. Check /schedule page for upcoming games"
    echo "  4. Verify daily updates run at 3:30 AM UTC"
else
    echo "⚠️  Some checks failed. Review output above."
    echo ""
    echo "Common fixes:"
    echo "  - Run: python3 live-system/populate_database_for_frontend.py"
    echo "  - Check Railway logs for errors"
    echo "  - Verify DATABASE_URL is correct"
fi

echo ""
echo "================================================================================"

