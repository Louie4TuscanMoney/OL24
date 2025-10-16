# NBA_API Setup

**Status:** Ready for implementation  
**Purpose:** Live NBA score streaming for ML model

---

## Quick Start

### 1. Install Dependencies
```bash
cd "Action/2. NBA API/1. API Setup"
pip install -r requirements.txt
```

### 2. Test Installation
```bash
python test_nba_api.py
```

### 3. Expected Output
```
✅ NBA_API imported successfully
✅ Fetched 30 NBA teams
✅ Live scoreboard fetched
✅ NBA_API INSTALLATION TEST PASSED
```

---

## What's Next

After installation test passes:
1. Build `live_score_buffer.py` - Minute-by-minute score accumulation
2. Build `ml_connector.py` - Connect buffer to ML model
3. Test end-to-end: Live game → Buffer → ML prediction

---

## Integration with ML Model

**ML Model Location:** `../1. ML/X. MVP Model/`

**ML Model expects:** 18-number array (minute-by-minute differentials)

**NBA_API provides:** Live scores every ~10 seconds

**Our job:** Convert real-time updates → 18-minute pattern

---

*Run test_nba_api.py first to verify installation*

