# ⚡ QUICK START GUIDE

Get started with the BetOnline scraper in 15 minutes.

---

## 🚀 INSTALLATION

### Step 1: Install Dependencies

```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/betonlineofficial"

# Install Python dependencies
pip install playwright playwright-stealth beautifulsoup4 requests

# Install Playwright browsers
python -m playwright install chromium
```

### Step 2: Verify Installation

```bash
python3 -c "from playwright.sync_api import sync_playwright; print('✅ Playwright installed')"
python3 -c "from playwright_stealth import stealth; print('✅ Stealth plugin installed')"
python3 -c "import requests; print('✅ Requests installed')"
```

---

## 🎯 BASIC USAGE

### Example 1: Simple Scraping

```python
from scrapers.betonline_live_lines import BetOnlineScraper

# Initialize scraper
scraper = BetOnlineScraper()

# Get live lines (tries multiple methods)
lines = scraper.get_live_lines()

# Display results
print(f"Found {len(lines)} live games:")
for line in lines:
    print(f"\n{line['away_team']} @ {line['home_team']}")
    print(f"  Spread: {line['spread']}")
    print(f"  Total: {line['total']}")
    print(f"  ML: {line['moneyline_home']} / {line['moneyline_away']}")
```

### Example 2: With Crawlee

```python
from scrapers.crawlee_betonline_scraper import get_crawlee_betonline_odds

# Get odds using Crawlee (browser automation)
odds = get_crawlee_betonline_odds()

if odds:
    print(f"✅ Crawlee: Found {len(odds)} games")
    for game in odds:
        print(game)
else:
    print("❌ Crawlee failed - check selectors!")
```

### Example 3: With Implied Probabilities

```python
from scrapers.betonline_live_lines import BetOnlineScraper
from utilities.implied_probability_calculator import ImpliedProbabilityCalculator

scraper = BetOnlineScraper()
calc = ImpliedProbabilityCalculator()

lines = scraper.get_live_lines()

for line in lines:
    # Calculate implied probabilities
    home_ml = line['moneyline_home']
    away_ml = line['moneyline_away']
    
    home_prob = calc.american_to_probability(home_ml)
    away_prob = calc.american_to_probability(away_ml)
    
    # Calculate no-vig probabilities
    no_vig = calc.no_vig_probability(home_ml, away_ml)
    vig = calc.vig_percentage(home_ml, away_ml)
    
    print(f"\n{line['away_team']} @ {line['home_team']}")
    print(f"  Implied: Home {home_prob:.1%} / Away {away_prob:.1%}")
    print(f"  No-Vig: Home {no_vig['home']:.1%} / Away {no_vig['away']:.1%}")
    print(f"  Vig: {vig:.2f}%")
```

---

## 🔧 CURRENT STATUS

### ⚠️ IMPORTANT: Scraper Needs Fixing!

The scraper is **NOT currently working** because it uses placeholder CSS selectors.

**What Works:**
- ✅ Infrastructure setup
- ✅ Playwright installed
- ✅ Stealth plugin integrated
- ✅ Implied probability calculator
- ✅ Multi-method fallback system

**What Doesn't Work:**
- ❌ Crawlee scraper returns 0 games
- ❌ Using placeholder selectors
- ❌ Falls back to synthetic odds

**To Fix:**
1. Read `documentation/WHAT_FAILED.md` (understand the problem)
2. Read `documentation/SOLUTIONS_GUIDE.md` (step-by-step fix)
3. Inspect BetOnline HTML (find real selectors)
4. Update `scrapers/crawlee_betonline_scraper.py` (replace placeholders)
5. Test and iterate

**Time to Fix:** 2-4 hours

---

## 📖 DOCUMENTATION

### Start Here:

1. **README.md** - Package overview and structure
2. **documentation/WHAT_FAILED.md** - Why scraper doesn't work
3. **documentation/SOLUTIONS_GUIDE.md** - How to fix it
4. **documentation/TECHNICAL_FINDINGS.md** - BetOnline architecture
5. **documentation/FUTURE_ROADMAP.md** - Long-term plan

### For Developers:

- **specifications/** - Original specs and requirements
- **examples/** - Working code examples
- **scrapers/** - Source code with comments
- **utilities/** - Helper functions

---

## 🎯 COMMON TASKS

### Task 1: Test if Scraper Works

```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/betonlineofficial"

python3 -c "
from scrapers.crawlee_betonline_scraper import get_crawlee_betonline_odds

print('🕷️ Testing Crawlee scraper...')
odds = get_crawlee_betonline_odds()

if odds and len(odds) > 0:
    print(f'✅ SUCCESS: Found {len(odds)} games!')
    for game in odds[:3]:
        print(f'  {game}')
else:
    print('❌ FAILED: Scraper not working yet')
    print('Follow SOLUTIONS_GUIDE.md to fix!')
"
```

### Task 2: Check Current Fallback (Synthetic Odds)

```bash
python3 -c "
from scrapers.betonline_live_lines import BetOnlineScraper

scraper = BetOnlineScraper()
lines = scraper.get_live_lines()

print(f'Found {len(lines)} games (using fallback)')
for line in lines:
    print(f'{line[\"away_team\"]} @ {line[\"home_team\"]}: {line[\"source\"]}')
"
```

### Task 3: Test Implied Probability Calculator

```bash
python3 -c "
from utilities.implied_probability_calculator import ImpliedProbabilityCalculator

calc = ImpliedProbabilityCalculator()

# Test American odds conversion
home_ml = -150
away_ml = 130

home_prob = calc.american_to_probability(home_ml)
away_prob = calc.american_to_probability(away_ml)

print(f'American Odds: {home_ml} / {away_ml}')
print(f'Implied Prob: {home_prob:.1%} / {away_prob:.1%}')

no_vig = calc.no_vig_probability(home_ml, away_ml)
vig = calc.vig_percentage(home_ml, away_ml)

print(f'No-Vig Prob: {no_vig[\"home\"]:.1%} / {no_vig[\"away\"]:.1%}')
print(f'Vig: {vig:.2f}%')
"
```

---

## 🐛 TROUBLESHOOTING

### Issue: `ModuleNotFoundError: No module named 'playwright'`

**Fix:**
```bash
pip install playwright playwright-stealth
python -m playwright install chromium
```

### Issue: `ImportError: cannot import name 'stealth_async'`

**Fix:**
```bash
# Stealth plugin might not be installed
pip install playwright-stealth

# OR it might be a different package
pip install playwright-stealth-py
```

### Issue: Scraper returns 0 games

**This is expected!** The scraper uses placeholder selectors.

**Fix:** Follow `documentation/SOLUTIONS_GUIDE.md` to:
1. Inspect BetOnline HTML
2. Find real CSS selectors
3. Update scraper code
4. Test and iterate

### Issue: 403 Forbidden errors

**This is anti-bot protection.**

**Fix:** Follow `documentation/SOLUTIONS_GUIDE.md` Section 2:
1. Improve stealth configuration
2. Add realistic behavior (delays, mouse movements)
3. Consider residential proxy
4. Consider ScraperAPI service

---

## 📚 LEARNING PATH

### Beginner: Just Want to Use It

1. **Install dependencies** (5 min)
2. **Run basic example** (5 min)
3. **Read WHAT_FAILED.md** (15 min)
4. **Understand the issue** (placeholder selectors)

**Time:** 30 minutes  
**Outcome:** Understand what needs to be fixed

### Intermediate: Want to Fix It

1. **Follow SOLUTIONS_GUIDE.md** (2-4 hours)
2. **Inspect BetOnline HTML** (30 min)
3. **Update selectors** (30 min)
4. **Test and iterate** (1-2 hours)

**Time:** 3-5 hours  
**Outcome:** Working scraper with real odds

### Advanced: Want Production System

1. **Complete Intermediate** (3-5 hours)
2. **Add error handling** (2-3 hours)
3. **Add monitoring** (2-3 hours)
4. **Optimize performance** (2-3 hours)
5. **Test for 1+ week** (passive monitoring)

**Time:** 10-15 hours + 1 week testing  
**Outcome:** Production-ready scraper with 95%+ uptime

---

## 🎓 NEXT STEPS

**If you're new:**
1. Read `README.md` (10 min)
2. Read `documentation/WHAT_FAILED.md` (20 min)
3. Understand the problem

**If you want to fix it:**
1. Read `documentation/SOLUTIONS_GUIDE.md` (30 min)
2. Follow Step 1: Fix Selectors (1-2 hours)
3. Test and iterate

**If you want production:**
1. Complete the fix (3-5 hours)
2. Read `documentation/FUTURE_ROADMAP.md` (30 min)
3. Follow Phase 2: Production Hardening (10-12 hours)

---

## 💡 TIPS

### Tip 1: Start Small
Don't try to fix everything at once. Fix selectors first, then add features.

### Tip 2: Test in Browser First
Always test selectors in browser console before writing code.

### Tip 3: Iterate and Debug
Expect to iterate 3-5 times before it works perfectly.

### Tip 4: Log Everything
Add `print()` statements liberally to understand what's happening.

### Tip 5: Take Screenshots
Use `await page.screenshot(path='debug.png')` to see what Playwright sees.

---

## 📞 SUPPORT

### Documentation:
- **README.md** - Overview
- **WHAT_FAILED.md** - Root causes
- **SOLUTIONS_GUIDE.md** - Step-by-step fixes
- **TECHNICAL_FINDINGS.md** - Deep dive
- **FUTURE_ROADMAP.md** - Long-term plan

### Code:
- **scrapers/** - Source code
- **utilities/** - Helper functions
- **examples/** - Working examples

---

**⚡ Get started in 15 minutes!**

**🔧 Fix in 2-4 hours!**

**🚀 Production in 2-3 weeks!**

