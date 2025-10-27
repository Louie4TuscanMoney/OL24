# 🚀 STRYK MIGRATION PACKAGE

**Complete Migration from Current Backend → STRYK Production**

---

## 🎯 **WHAT IS THIS?**

This package contains **everything** you need to migrate your current NBA trading system to a production-ready STRYK deployment with:

- **PostgreSQL** - Production database (Railway/Render)
- **Redis** - High-performance caching
- **FastAPI** - Optimized backend
- **SolidJS** - Modern frontend
- **Zero downtime** - Seamless migration
- **100% data integrity** - All predictions preserved

---

## 📦 **PACKAGE STRUCTURE**

```
STRYK_MIGRATION/
├── schemas/
│   └── postgres_schema.sql         # Complete Postgres schema
│
├── scripts/
│   ├── 01_export_old_data.py       # Export from current backend
│   ├── 02_migrate_to_stryk.py      # Migrate to Postgres
│   ├── 03_warmup_redis.py          # Warm Redis cache
│   ├── 04_deploy_backend.sh        # Deploy to Railway
│   └── 05_deploy_frontend.sh       # Deploy to Vercel
│
├── backend/
│   └── main.py                     # Updated FastAPI with Postgres + Redis
│
├── export/
│   └── migration_data.json         # (Generated) Exported data
│
├── documentation/
│   ├── MIGRATION_GUIDE.md          # Step-by-step guide
│   ├── ROLLBACK_PLAN.md            # How to rollback if needed
│   └── TROUBLESHOOTING.md          # Common issues
│
└── README.md                       # This file
```

---

## ⚡ **QUICK START (10-STEP MIGRATION)**

### **Prerequisites**

```bash
# Install dependencies
pip install httpx sqlalchemy redis psycopg2-binary

# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login
```

---

### **STEP 1: Deploy Postgres + Redis on Railway** *(10 min)*

```bash
# Create Railway project
railway init  # Name: stryk-prod

# Add Postgres
railway add postgres

# Add Redis
railway add redis
```

**Output**: You'll get `POSTGRES_URL` and `REDIS_URL`

**Save them**:
```bash
# .env.local
POSTGRES_URL=postgresql://...
REDIS_URL=redis://...
```

---

### **STEP 2: Create Postgres Schema** *(5 min)*

```bash
# Connect to Railway Postgres
railway run psql $POSTGRES_URL

# Paste schema (or run file)
\i schemas/postgres_schema.sql

# Verify
\dt
```

**Expected output**: 6 tables (games, predictions, mamba_scores, users, bets, user_requests)

---

### **STEP 3: Export Current Backend Data** *(10 min)*

```bash
cd scripts

# Update OLD_BACKEND_URL in 01_export_old_data.py
# Change line 19 to your backend URL:
OLD_BACKEND_URL = "http://localhost:8001"  # or your ngrok URL

# Run export
python 01_export_old_data.py
```

**Output**: `../export/migration_data.json`

---

### **STEP 4: Migrate to Postgres** *(15 min)*

```bash
# Set environment variables
export POSTGRES_URL="postgresql://..."  # From Railway

# Run migration
python 02_migrate_to_stryk.py
```

**Output**: All games, predictions, and Mamba scores migrated to Postgres

---

### **STEP 5: Warm Redis Cache** *(5 min)*

```bash
# Set environment variables
export REDIS_URL="redis://..."  # From Railway

# Run warmup
python 03_warmup_redis.py
```

**Output**: Active games, features, predictions cached in Redis

---

### **STEP 6: Update Backend** *(15 min)*

```bash
# Copy backend template
cp backend/main.py /path/to/your/stryk/backend/

# Update imports in your backend
# Add Postgres + Redis connections (see backend/main.py)
```

**Key changes**:
- Replace in-memory storage with Postgres queries
- Add Redis caching layer
- Use connection pooling
- Add health check endpoint

---

### **STEP 7: Deploy Backend to Railway** *(10 min)*

```bash
cd /path/to/your/stryk/backend

# Initialize git if not already
git init
git add .
git commit -m "STRYK: Postgres + Redis backend"

# Deploy to Railway
railway up

# Set environment variables in Railway dashboard
railway variables set POSTGRES_URL=$POSTGRES_URL
railway variables set REDIS_URL=$REDIS_URL
```

**Output**: Backend running at `https://your-app.railway.app`

---

### **STEP 8: Deploy Frontend to Vercel** *(10 min)*

```bash
cd /path/to/your/stryk/frontend

# Update API URL in .env.production
VITE_API_BASE_URL=https://your-app.railway.app

# Deploy
vercel --prod
```

**Output**: Frontend live at `strykbig.ai` (or your domain)

---

### **STEP 9: Verify Migration** *(5 min)*

```bash
# Test backend
curl https://your-app.railway.app/api/live-games

# Test frontend
open https://strykbig.ai

# Check Railway logs
railway logs
```

**Expected**: Live games visible, predictions accurate, no errors

---

### **STEP 10: Go Live** *(5 min)*

```bash
# Update DNS (if custom domain)
# Point strykbig.ai → Vercel

# Monitor Railway dashboard
railway dashboard

# Celebrate 🎉
echo "STRYK IS LIVE!"
```

---

## 🗂️ **POSTGRES SCHEMA OVERVIEW**

### **Core Tables**

#### **1. `games` - Live Game Data**
```sql
- id (UUID, primary key)
- external_id (text, unique) -- NBA game ID
- home_team, away_team (text)
- home_score, away_score (int)
- status (int) -- 1=scheduled, 2=live, 3=final
- period, clock (int, text)
- spread, total, home_ml, away_ml (decimal, int)
- spread_display, underdog_display (text) -- "Lakers -6.0"
- implied probabilities, vig (decimal)
- is_q2_6min, can_predict (boolean)
```

#### **2. `predictions` - ML Predictions**
```sql
- id (UUID, primary key)
- game_id (UUID, foreign key → games)
- model_version (text) -- "mamba_v1"
- spread_pred, mamba_score (decimal)
- mamba_probability, market_probability (decimal)
- kelly_edge, kelly_bet_size (decimal)
- recommended_bet, bet_confidence (text)
- archetype, risk_score (text, decimal)
- features (jsonb) -- 67 Mamba features
```

#### **3. `mamba_scores` - Stored Scores (Q2 6:00+)**
```sql
- id (UUID, primary key)
- game_id (UUID, foreign key → games)
- external_id, home_team, away_team (text)
- mamba_score (decimal)
- period, clock (int, text)
```

#### **4. `users` - User Management**
```sql
- id (UUID, primary key)
- username, email (text, unique)
- bankroll, total_bet, total_profit, roi (decimal)
- status (text) -- "pending", "approved", "banned"
```

#### **5. `bets` - Bet Tracking**
```sql
- id (UUID, primary key)
- user_id (UUID, foreign key → users)
- game_id (UUID, foreign key → games)
- bet_type, bet_side (text) -- "spread", "home"
- stake, odds, potential_payout (decimal, int, decimal)
- status (text) -- "pending", "won", "lost", "push"
```

#### **6. `user_requests` - Access Requests**
```sql
- id (UUID, primary key)
- username, email, reason (text)
- status (text) -- "pending", "approved", "rejected"
- admin_notes (text)
```

### **Indexes**
- `idx_games_status` - Fast live game queries
- `idx_games_period` - Q2/Q3/Q4 filtering
- `idx_predictions_kelly_edge` - Top opportunities
- All foreign keys indexed

### **Views**
- `live_opportunities` - Active betting opportunities (status=2, period>=2, edge>2%)
- `user_performance` - User stats (total bets, wins, losses, ROI)

---

## 🔄 **REDIS CACHING STRATEGY**

### **Cache Keys**

```
live:games:active          SET     Active game IDs
features:nba:{game_id}     STRING  67 Mamba features (TTL: 24h)
prediction:nba:{game_id}   STRING  ML prediction (TTL: 1h)
mamba_score:nba:{game_id}  STRING  Stored Mamba score (TTL: 24h)
user:session:{user_id}     STRING  User session (TTL: 7d)
```

### **Cache Invalidation**

- **Features**: Regenerate on Q2 6:00 mark
- **Predictions**: Regenerate every 30 seconds (live games)
- **Mamba scores**: Persist for 24 hours (historical)
- **Sessions**: 7-day rolling expiry

---

## 📊 **MIGRATION CHECKLIST**

```
☐ Step 1: Deploy Postgres + Redis on Railway (10 min)
☐ Step 2: Create Postgres schema (5 min)
☐ Step 3: Export current backend data (10 min)
☐ Step 4: Migrate to Postgres (15 min)
☐ Step 5: Warm Redis cache (5 min)
☐ Step 6: Update backend code (15 min)
☐ Step 7: Deploy backend to Railway (10 min)
☐ Step 8: Deploy frontend to Vercel (10 min)
☐ Step 9: Verify migration (5 min)
☐ Step 10: Go live (5 min)

TOTAL: ~2 hours
```

---

## 🚨 **ROLLBACK PLAN**

If something goes wrong, you can **rollback** instantly:

### **Quick Rollback**

```bash
# 1. Point frontend back to old backend
cd frontend
vercel env add VITE_API_BASE_URL http://your-old-backend.com
vercel --prod

# 2. Keep old backend running until migration verified
# Don't shut down old backend for 24 hours

# 3. If needed, restore from export
python scripts/restore_from_export.py
```

### **Data Preservation**

- **Export file** (`migration_data.json`) - Keep for 30 days
- **Old backend** - Keep running for 24 hours
- **Postgres backups** - Railway auto-backups daily
- **Redis persistence** - Enable RDB snapshots

---

## 🔧 **TROUBLESHOOTING**

### **Issue: Export fails with connection error**

```bash
# Check backend is running
curl http://localhost:8001/api/live-games

# Use ngrok if backend is local
ngrok http 8001
# Update OLD_BACKEND_URL to ngrok URL
```

### **Issue: Postgres migration fails**

```bash
# Check Postgres URL is correct
echo $POSTGRES_URL

# Test connection
railway run psql $POSTGRES_URL

# Check schema exists
\dt

# Re-run schema if needed
\i schemas/postgres_schema.sql
```

### **Issue: Redis connection fails**

```bash
# Check Redis URL is correct
echo $REDIS_URL

# Test connection
redis-cli -u $REDIS_URL
PING  # Should return PONG

# Restart Redis if needed
railway restart redis
```

### **Issue: Railway deployment fails**

```bash
# Check Railway logs
railway logs

# Common fixes:
# - Add requirements.txt with all dependencies
# - Add Procfile: web: uvicorn main:app --host 0.0.0.0 --port $PORT
# - Set environment variables in Railway dashboard
```

---

## 📈 **POST-MIGRATION OPTIMIZATION**

### **Week 1: Monitoring**

```bash
# Add monitoring
railway add plugin sentry    # Error tracking
railway add plugin logtail   # Log aggregation

# Set alerts
- Email on DB failure
- Slack on high latency (>500ms)
- Discord on prediction errors
```

### **Week 2: Performance**

```bash
# Optimize queries
CREATE INDEX idx_custom ON predictions(kelly_edge DESC) WHERE kelly_edge > 0.02;

# Add read replicas (Railway Pro)
railway add postgres-replica

# Upgrade Redis (Railway Pro)
railway upgrade redis  # More memory
```

### **Week 3: Features**

```bash
# Add user authentication
# Add bankroll management
# Add bet tracking
# Add admin dashboard
```

---

## 💰 **COST BREAKDOWN**

### **Railway Free Tier**
- **Postgres**: 5GB storage, 1GB RAM (Free)
- **Redis**: 256MB RAM (Free)
- **Backend**: 500 hours/month (Free)
- **Total**: **$0/month**

### **Railway Pro** *(Recommended for production)*
- **Postgres**: 10GB storage, 2GB RAM ($10/month)
- **Redis**: 1GB RAM ($5/month)
- **Backend**: Unlimited hours ($20/month)
- **Total**: **$35/month**

### **Vercel** *(Frontend)*
- **Free**: 100GB bandwidth (Free)
- **Pro**: Unlimited bandwidth ($20/month)

### **Total Monthly Cost**
- **Hobby**: $0/month (Railway Free + Vercel Free)
- **Production**: $35-55/month (Railway Pro + Vercel Pro)

---

## 📚 **DOCUMENTATION**

- **[MIGRATION_GUIDE.md](documentation/MIGRATION_GUIDE.md)** - Detailed step-by-step guide
- **[ROLLBACK_PLAN.md](documentation/ROLLBACK_PLAN.md)** - How to rollback safely
- **[TROUBLESHOOTING.md](documentation/TROUBLESHOOTING.md)** - Common issues and fixes

---

## 🎯 **NEXT STEPS AFTER MIGRATION**

1. **Add user authentication** (JWT tokens, session management)
2. **Build admin dashboard** (approve users, view bets, monitor system)
3. **Add multi-sport support** (NFL, MLB, NHL schemas)
4. **Enable payments** (Stripe integration, subscription tiers)
5. **Advanced risk management** (portfolio optimization, delta tracking)

---

## 🤝 **SUPPORT**

- **Discord**: strykbig.ai/discord
- **Email**: support@strykbig.ai
- **X**: @strykbig

---

**🚀 STRYK: Your backend data → Production-ready in 2 hours! 🚀**

**Hercules. Lightning. Dollar. Edge.**

