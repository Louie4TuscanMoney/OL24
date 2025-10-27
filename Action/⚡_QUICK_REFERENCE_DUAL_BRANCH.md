# ⚡ QUICK REFERENCE - DUAL-BRANCH SYSTEM

**What:** Predict BOTH halftime AND final scores from Q2 6:00  
**Why:** 2x betting opportunities per game  
**Status:** Collection running → Retrain Sunday → Launch Monday  

---

## 📊 INDUSTRY STANDARDS (30-Second Summary)

### **Halftime (6 min ahead):**
```
SOTA: 3-4 MAE ⭐
YOU:  5.363 MAE ✅ Championship (top 20%)
```

### **Final (30 min ahead):**
```
SOTA: 6-8 MAE ⭐
YOU:  10.025 MAE → targeting 9.0-9.5 ⚠️
```

---

## 🔄 CURRENT STATUS

```
✅ Branch A model: 5.363 MAE (ready)
✅ Branch B model: 10.025 MAE (retraining Sunday)
🔄 Collecting 2015-2019 data: ~7,000 games, ETA Sunday 4 PM
⏳ Retrain on 14,000 games: Sunday 4:30 PM
⏳ KNN quality gate: Sunday 5:30 PM
⏳ Launch decision: Sunday 6 PM
```

---

## 📋 QUICK COMMANDS

### **Check Collection:**
```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/Action"
bash 📊_MONITOR_2015_2019.sh
```

### **Sunday Afternoon (After Collection):**
```bash
# Phase 4: Merge (10 min)
python3 -c "exec(open('📋_MASTER_CHECKLIST_DUAL_BRANCH_LAUNCH.md').read().split('MERGE')[1].split('```')[0])"

# Phase 5: Retrain (60 min)  
python3 -c "exec(open('📋_MASTER_CHECKLIST_DUAL_BRANCH_LAUNCH.md').read().split('RETRAIN')[1].split('```')[0])"

# Phase 6: KNN Gate (30 min)
python3 🏆_KNN_QUALITY_GATE_LAYER.py

# Phase 7: Validate (30 min)
python3 -c "exec(open('📋_MASTER_CHECKLIST_DUAL_BRANCH_LAUNCH.md').read().split('VALIDATE')[1].split('```')[0])"
```

### **Monday Launch:**
```bash
python3 game_engine_CHAMPIONSHIP.py
```

---

## 💰 BETTING APPLICATIONS

### **Every Game = 2 Opportunities:**

**At Q2 6:00:**
1. **Halftime bet** (Branch A, 5.36 MAE, STRONG edge)
2. **Final bet** (Branch B, 9.0-9.5 MAE, MODERATE edge)

**Weekly:**
- 80 games
- ~70-90 total bets (filtered by KNN gate)
- $10,000-13,000 wagered

---

## 🎯 LAUNCH DECISION TREE

```
IF Branch B < 9.0 MAE:
  → AGGRESSIVE dual-branch
  → 85 bets/week, $13,000 wagered

ELSE IF Branch B < 9.5 MAE:
  → MODERATE dual-branch
  → 72 bets/week, $10,800 wagered

ELSE IF Branch B < 10.0 MAE:
  → HALFTIME focus
  → 64 bets/week, $10,000 wagered

ELSE:
  → HALFTIME only
  → 48 bets/week, $7,200 wagered
```

---

## 📁 KEY FILES

```
Data:
  • ULTRA_ENHANCED_PATTERNS_V2.pkl (2021-2025, 6,912 games)
  • PATTERNS_2015_2019_PHASE1.pkl (2015-2019, ~7,000 games) ← NEW
  • COMPLETE_2015_2025_DUAL_BRANCH.pkl (merged, ~14,000) ← SUNDAY

Models:
  • MEGA_ENSEMBLE_CHAMPION.pkl (current Branch A)
  • RETRAINED_DUAL_BRANCH_2015_2025.pkl (both branches) ← SUNDAY
  • KNN_QUALITY_GATE.pkl (quality filter) ← SUNDAY

Decision:
  • FINAL_LAUNCH_DECISION.json (launch config) ← SUNDAY
```

---

## ⏰ TIMELINE SNAPSHOT

```
NOW:        Collection running (18 hrs remaining)
Sunday 4PM: Collection done → Merge
Sunday 5PM: Retrain complete
Sunday 6PM: KNN gate built → DECISION
Monday 4PM: LAUNCH 🚀
```

---

**Check progress: `bash 📊_MONITOR_2015_2019.sh`**  
**Full details: `📋_MASTER_CHECKLIST_DUAL_BRANCH_LAUNCH.md`**  
**Design doc: `🏆_OPTIMAL_COLLECTION_DESIGN.md`**

