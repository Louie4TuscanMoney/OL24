# 📊 DAILY MAMBA REVIEW GUIDE

**Automatic Logging System** - Review Mamba predictions every day!

---

## **🎯 WHAT'S AUTOMATICALLY LOGGED:**

Every time Mamba makes a prediction, the system automatically stores:

✅ **Prediction Details:**
- Mamba prediction value
- Game state (scores, period, clock)
- BetOnline odds
- Edge and confidence

✅ **Live Game Data:**
- Home/away teams
- Current scores
- Period and clock
- Game status

✅ **Odds & Probability:**
- Spread, total, moneylines
- Implied probabilities
- Vig percentage
- Odds source

✅ **Post-Game Outcome:**
- Final scores
- Actual spread
- Prediction accuracy (MAE)
- Bet results (if placed)

---

## **📂 WHERE LOGS ARE STORED:**

All logs are stored in: `/live-system/mamba_logs/`

```
mamba_logs/
├── daily/              # Daily prediction logs
│   ├── 2025-10-27.json
│   ├── 2025-10-28.json
│   └── ...
├── games/              # Game-specific logs
│   ├── 0022500123.json
│   ├── 0022500124.json
│   └── ...
└── summaries/          # Daily summaries
    ├── 2025-10-27_summary.json
    ├── 2025-10-28_summary.json
    └── ...
```

---

## **🔍 HOW TO REVIEW DAILY:**

### **Method 1: API Endpoint (Easiest)**

```bash
# Get today's summary
curl https://ol24-production.up.railway.app/api/mamba-daily-log

# Get specific date
curl "https://ol24-production.up.railway.app/api/mamba-daily-log?date=2025-10-27"
```

**Response:**
```json
{
  "daily_summary": {
    "date": "2025-10-27",
    "total_predictions": 5,
    "completed": 5,
    "pending": 0,
    "accuracy": {
      "average_mae": 8.2,
      "within_5pts": 2,
      "within_10pts": 4,
      "within_5pts_pct": 0.4,
      "within_10pts_pct": 0.8
    },
    "betting": {
      "bets_placed": 3,
      "wins": 2,
      "losses": 1,
      "win_rate": 0.667,
      "total_wagered": 300.00,
      "total_profit": 82.00,
      "roi": 0.273
    }
  }
}
```

### **Method 2: Python Script**

```python
from mamba_auto_logger import MambaAutoLogger

logger = MambaAutoLogger()
logger.print_daily_report()  # Prints formatted report
```

### **Method 3: Direct JSON Files**

```bash
# View today's predictions
cat mamba_logs/daily/$(date +%Y-%m-%d).json | jq '.'

# View specific game
cat mamba_logs/games/0022500123.json | jq '.'

# View today's summary
cat mamba_logs/summaries/$(date +%Y-%m-%d)_summary.json | jq '.'
```

---

## **📋 DAILY REVIEW CHECKLIST:**

Every day, check:

### **1. Total Predictions** ✅
- How many predictions were made today?
- Which games triggered predictions?

### **2. Accuracy** ✅
- Average MAE (target: <10 points)
- % within 5 points (target: >30%)
- % within 10 points (target: >65%)

### **3. Betting Performance** ✅
- How many bets were placed?
- Win rate (target: >55%)
- Total profit/loss
- ROI (target: >10%)

### **4. Edge Analysis** ✅
- Were edges ≥5 points?
- Did high-edge bets perform better?
- Any patterns in wins/losses?

### **5. Game Types** ✅
- Which game types were predicted?
- Did "Lead Held" games perform best?
- Any surprises in "Close" games?

---

## **🎯 HOW TO UPDATE OUTCOMES:**

### **Automatic (Via API):**

```bash
curl -X POST https://ol24-production.up.railway.app/api/mamba-update-outcome \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "0022500123",
    "final_home_score": 112,
    "final_away_score": 108,
    "bet_placed": true,
    "bet_amount": 100,
    "bet_result": "win"
  }'
```

### **Manual (Python Script):**

```python
from mamba_auto_logger import MambaAutoLogger

logger = MambaAutoLogger()

# Update outcome
logger.update_outcome(
    game_id="0022500123",
    final_home_score=112,
    final_away_score=108,
    bet_placed=True,
    bet_amount=100,
    bet_result="win"  # or "loss" or "push"
)
```

---

## **📊 EXAMPLE DAILY REPORT:**

```
================================================================================
📊 MAMBA DAILY REPORT - 2025-10-27
================================================================================

Total Predictions: 5
Completed: 5
Pending: 0

📈 Accuracy:
  Average MAE: 8.20 points
  Within 5pts: 2 (40.0%)
  Within 10pts: 4 (80.0%)

💰 Betting:
  Bets Placed: 3
  Wins: 2
  Losses: 1
  Win Rate: 66.7%
  Total Wagered: $300.00
  Total Profit: $82.00
  ROI: 27.3%

================================================================================
```

---

## **🎓 WHAT TO LOOK FOR:**

### **✅ Good Signs:**
- MAE < 10 points
- >50% of predictions within 10 points
- Win rate > 55%
- ROI > 10%
- High-confidence bets winning more

### **⚠️ Warning Signs:**
- MAE > 12 points
- Win rate < 50%
- Negative ROI
- High-edge bets losing consistently

### **📈 Improvement Actions:**
- Increase edge threshold if MAE is high
- Only bet on "Lead Held" games
- Adjust Kelly Criterion fraction
- Review feature extraction quality

---

## **🔔 DAILY ROUTINE:**

### **Morning After Games:**
1. Check yesterday's log: `curl https://ol24-production.up.railway.app/api/mamba-daily-log`
2. Review accuracy and betting performance
3. Update outcomes if not auto-updated
4. Note any patterns or anomalies

### **Before Next Games:**
1. Review previous day's learnings
2. Adjust betting strategy if needed
3. Check if model needs retraining
4. Verify system is ready for new predictions

---

## **💡 PRO TIPS:**

### **Track Trends:**
```bash
# Get last 7 days of summaries
for i in {0..6}; do
  date=$(date -v-${i}d +%Y-%m-%d 2>/dev/null || date -d "$i days ago" +%Y-%m-%d)
  echo "=== $date ==="
  curl -s "https://ol24-production.up.railway.app/api/mamba-daily-log?date=$date" | jq '.daily_summary.accuracy.average_mae'
done
```

### **Compare to Baseline:**
- MAE should be close to training MAE (9.029 points)
- Win rate should exceed 55% (to beat -110 vig)
- ROI should be positive over 30+ bets

### **Watch for Drift:**
- If MAE increases over time → Model needs retraining
- If specific teams perform poorly → Add team-specific adjustments
- If certain periods are weak → Improve feature extraction

---

## **📞 QUICK ACCESS:**

### **View Today's Log:**
```bash
curl https://ol24-production.up.railway.app/api/mamba-daily-log
```

### **View Specific Game:**
```bash
# Find game ID from logs, then:
cat mamba_logs/games/0022500123.json | jq '.'
```

### **Print Report:**
```bash
cd live-system
python -c "from mamba_auto_logger import MambaAutoLogger; MambaAutoLogger().print_daily_report()"
```

---

## **✅ SUMMARY:**

✅ **Every prediction is automatically logged**
✅ **View daily summaries via API**
✅ **Track accuracy, betting, and ROI**
✅ **Review every morning after games**
✅ **Stored in `mamba_logs/` directory**

**Your Mamba system now has full accountability and transparency!** 🎯📊🚀


