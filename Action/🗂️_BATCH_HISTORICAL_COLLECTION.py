#!/usr/bin/env python3
"""
🗂️ BATCH HISTORICAL DATA COLLECTION

Collects NBA game data in 5-year batches going back to 2005
Stores both detailed patterns and executive summary CSVs

BATCHES:
- 2020-2025: Current era (already running!)
- 2015-2020: Warriors dynasty, pace-and-space
- 2010-2015: Early analytics, 3PT revolution
- 2005-2010: Post-hand-check, ISO ball

OUTPUT:
1. Detailed patterns (70 features per game) - For ML training
2. Executive summary CSV - For business stakeholders
3. Combined dataset - For ensemble training

DESIGN: Queue system - runs one batch at a time
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle

class BatchCollectionManager:
    """
    Manages multi-year batch data collection
    """
    
    def __init__(self):
        self.batches = [
            {
                'name': '2020-2025',
                'seasons': ['22020', '22021', '22022', '22023', '22024'],
                'status': 'in_progress',  # Currently running!
                'description': 'Current era - extreme 3PT, positionless'
            },
            {
                'name': '2015-2020',
                'seasons': ['22015', '22016', '22017', '22018', '22019'],
                'status': 'queued',
                'description': 'Warriors dynasty - pace and space revolution'
            },
            {
                'name': '2010-2015',
                'seasons': ['22010', '22011', '22012', '22013', '22014'],
                'status': 'queued',
                'description': 'Early analytics - 3PT revolution begins'
            },
            {
                'name': '2005-2010',
                'seasons': ['22005', '22006', '22007', '22008', '22009'],
                'status': 'queued',
                'description': 'Post-hand-check era - ISO ball dominant'
            }
        ]
        
        self.output_dir = Path('historical_data_batches')
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_batch_collection_script(self, batch):
        """
        Generate collection script for a specific batch
        """
        script = f"""#!/usr/bin/env python3
'''
Collect data for {batch['name']} era
{batch['description']}
'''

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

print("Collecting {batch['name']} season data...")

seasons = {batch['seasons']}
all_games = []

for season in seasons:
    print(f"  Season {{season}}...")
    
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable='Regular Season'
    )
    
    games = gamefinder.get_data_frames()[0]
    all_games.append(games)

# Combine all seasons
df = pd.concat(all_games, ignore_index=True)

# Save
output_file = 'historical_games_{batch['name'].replace('-', '_')}_basic.csv'
df.to_csv(output_file, index=False)

print(f"✅ Collected {{len(df):,}} game records")
print(f"   Saved to: {{output_file}}")
"""
        
        script_path = self.output_dir / f"collect_{batch['name'].replace('-', '_')}.py"
        with open(script_path, 'w') as f:
            f.write(script)
        
        script_path.chmod(0o755)
        
        return script_path
    
    def create_executive_summary(self, pattern_files):
        """
        Create executive summary CSV from detailed patterns
        
        Format:
        - Season
        - Games count
        - Avg point differential
        - Avg volatility
        - Quality grade distribution
        - Prediction difficulty score
        """
        
        summaries = []
        
        for pattern_file in pattern_files:
            with open(pattern_file, 'rb') as f:
                patterns = pickle.load(f)
            
            # Extract summary statistics
            season = pattern_file.stem.split('_')[2]  # Extract from filename
            
            differentials = [p['diff_at_final'] for p in patterns if p['diff_at_final'] is not None]
            volatilities = [p['pattern_statistical']['volatility'] for p in patterns]
            quality_grades = [p['quality_metrics']['quality_grade'] for p in patterns]
            
            summary = {
                'season': season,
                'games_count': len(patterns),
                'avg_point_differential': np.mean(np.abs(differentials)),
                'std_point_differential': np.std(differentials),
                'avg_volatility': np.mean(volatilities),
                'pct_quality_a': sum(1 for g in quality_grades if g == 'A') / len(quality_grades) * 100,
                'pct_quality_b': sum(1 for g in quality_grades if g == 'B') / len(quality_grades) * 100,
                'pct_quality_c': sum(1 for g in quality_grades if g == 'C') / len(quality_grades) * 100,
                'prediction_difficulty': np.mean(volatilities) * np.mean(np.abs(differentials)) / 100,
                'total_features': 70,
                'collection_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            summaries.append(summary)
        
        # Create DataFrame
        summary_df = pd.DataFrame(summaries)
        
        # Sort by season
        summary_df = summary_df.sort_values('season')
        
        # Save executive summary
        output_file = self.output_dir / 'EXECUTIVE_SUMMARY.csv'
        summary_df.to_csv(output_file, index=False)
        
        print(f"\n📊 Executive Summary Created:")
        print(f"   File: {output_file}")
        print(f"   Seasons: {len(summary_df)}")
        print(f"\n{summary_df.to_string()}")
        
        return summary_df
    
    def create_queue_script(self):
        """
        Create master script that runs all batches in sequence
        """
        queue_script = """#!/bin/bash
# Master queue script - runs all batch collections sequentially

echo "="
echo "🗂️  BATCH HISTORICAL DATA COLLECTION QUEUE"
echo "="

# Batch 1: 2020-2025 (ALREADY RUNNING - skip!)
echo ""
echo "Batch 1: 2020-2025 ✅ IN PROGRESS"
echo "  (Currently running in main extraction)"
echo ""

# Batch 2: 2015-2020
echo "Batch 2: 2015-2020"
echo "  Collecting game IDs..."
python3 historical_data_batches/collect_2015_2020.py

echo "  Extracting patterns..."
python3 🚀_ULTRA_OPTIMIZED_EXTRACTION.py --input historical_games_2015_2020_basic.csv --output PATTERNS_2015_2020.pkl

# Batch 3: 2010-2015
echo ""
echo "Batch 3: 2010-2015"
echo "  Collecting game IDs..."
python3 historical_data_batches/collect_2010_2015.py

echo "  Extracting patterns..."
python3 🚀_ULTRA_OPTIMIZED_EXTRACTION.py --input historical_games_2010_2015_basic.csv --output PATTERNS_2010_2015.pkl

# Batch 4: 2005-2010
echo ""
echo "Batch 4: 2005-2010"
echo "  Collecting game IDs..."
python3 historical_data_batches/collect_2005_2010.py

echo "  Extracting patterns..."
python3 🚀_ULTRA_OPTIMIZED_EXTRACTION.py --input historical_games_2005_2010_basic.csv --output PATTERNS_2005_2010.pkl

# Generate executive summary
echo ""
echo "📊 Generating executive summary..."
python3 -c "
from 🗂️_BATCH_HISTORICAL_COLLECTION import BatchCollectionManager
import glob

manager = BatchCollectionManager()
pattern_files = glob.glob('PATTERNS_*.pkl')
manager.create_executive_summary(pattern_files)
"

echo ""
echo "✅ ALL BATCHES COMPLETE!"
echo "   Total collection time: ~8-10 hours"
echo "   Total games: ~40,000+"
echo "   Total features: ~2.8 million"
"""
        
        queue_path = Path('RUN_BATCH_QUEUE.sh')
        with open(queue_path, 'w') as f:
            f.write(queue_script)
        
        queue_path.chmod(0o755)
        
        return queue_path
    
    def print_status(self):
        """Print current collection status"""
        print("="*80)
        print("🗂️  BATCH COLLECTION STATUS")
        print("="*80)
        
        for i, batch in enumerate(self.batches, 1):
            status_emoji = {
                'in_progress': '⏳',
                'queued': '📋',
                'complete': '✅',
                'failed': '❌'
            }
            
            emoji = status_emoji.get(batch['status'], '❓')
            
            print(f"\nBatch {i}: {batch['name']} {emoji}")
            print(f"  Status: {batch['status'].upper()}")
            print(f"  Description: {batch['description']}")
            print(f"  Seasons: {', '.join(batch['seasons'])}")


# Main execution
if __name__ == "__main__":
    manager = BatchCollectionManager()
    
    print("="*80)
    print("🗂️  BATCH HISTORICAL DATA COLLECTION SETUP")
    print("="*80)
    
    # Show current status
    manager.print_status()
    
    # Generate collection scripts for each batch
    print("\n" + "="*80)
    print("📝 GENERATING COLLECTION SCRIPTS")
    print("="*80)
    
    for batch in manager.batches[1:]:  # Skip first (already running)
        script_path = manager.generate_batch_collection_script(batch)
        print(f"\n✅ Generated: {script_path}")
    
    # Create master queue script
    print("\n" + "="*80)
    print("🎯 CREATING MASTER QUEUE SCRIPT")
    print("="*80)
    
    queue_path = manager.create_queue_script()
    print(f"\n✅ Created: {queue_path}")
    
    # Instructions
    print("\n" + "="*80)
    print("📋 NEXT STEPS")
    print("="*80)
    
    print(f"""
Current Status:
  ✅ Batch 1 (2020-2025): IN PROGRESS (finishes ~12:30 PM today)

To collect remaining batches:

Option 1: Run all batches sequentially (8-10 hours total)
  bash RUN_BATCH_QUEUE.sh

Option 2: Run batches individually
  # After current extraction finishes:
  python3 historical_data_batches/collect_2015_2020.py
  python3 🚀_ULTRA_OPTIMIZED_EXTRACTION.py  # Will auto-detect new file
  
  # Then:
  python3 historical_data_batches/collect_2010_2015.py
  python3 🚀_ULTRA_OPTIMIZED_EXTRACTION.py
  
  # Finally:
  python3 historical_data_batches/collect_2005_2010.py
  python3 🚀_ULTRA_OPTIMIZED_EXTRACTION.py

Timeline:
  - Current batch (2020-2025): Finishes 12:30 PM today
  - Each additional batch: ~2-3 hours
  - Total for all 4 batches: ~10-12 hours
  - Recommended: Run overnight

Output:
  - Detailed patterns: PATTERNS_YYYY_YYYY.pkl (70 features per game)
  - Executive summary: EXECUTIVE_SUMMARY.csv (for stakeholders)
  - Combined dataset: ~40,000 games, 20 years of NBA evolution

Next: Read 🎓_STANFORD_TEMPORAL_WEIGHTING.md for how to use this data!
""")
    
    print("="*80)

