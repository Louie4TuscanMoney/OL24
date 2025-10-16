"""
Test 05: Convert ALL 6 seasons to minute-by-minute time series
Process ~6,600 games → 48-point time series each
Following: Dejavu research + 02_DATA_PROCESSING.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

def convert_game_to_timeseries(game_df):
    """
    Convert single game to 48-minute time series
    Returns: array of 48 differentials
    """
    # Calculate game time
    game_df['GameTime'] = (game_df['Quarter'] - 1) * 720 + (720 - game_df['SecLeft'])
    
    differentials = []
    
    for minute in range(48):
        target_time = minute * 60
        plays_before = game_df[game_df['GameTime'] <= target_time]
        
        if len(plays_before) > 0:
            closest_play = plays_before.iloc[-1]
            differential = closest_play['HomeScore'] - closest_play['AwayScore']
        else:
            differential = 0  # Game start
        
        differentials.append(differential)
    
    return np.array(differentials)


def process_all_seasons():
    """
    Process all 6 NBA seasons
    """
    print("="*80)
    print("PROCESSING ALL 6 SEASONS → TIME SERIES")
    print("="*80)
    
    data_dir = Path("../../All Of Our Data")
    csv_files = sorted(data_dir.glob("NBA_PBP_*.csv"))
    
    print(f"\nFound {len(csv_files)} CSV files:")
    for f in csv_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  • {f.name} ({size_mb:.1f} MB)")
    
    print(f"\n⏱️  This will take 5-10 minutes...")
    input("Press Enter to start processing...")
    
    start_time = time.time()
    
    all_games_data = []
    total_plays = 0
    errors = 0
    
    for csv_file in csv_files:
        season = csv_file.stem.replace('NBA_PBP_', '')
        print(f"\n📂 Processing {season}...")
        
        # Load season
        df = pd.read_csv(csv_file)
        total_plays += len(df)
        print(f"   Plays: {len(df):,}")
        
        # Process each game
        games_in_season = df['URL'].nunique()
        print(f"   Games: {games_in_season:,}")
        
        games_processed = 0
        
        for game_url in df['URL'].unique():
            try:
                game_df = df[df['URL'] == game_url].copy()
                
                # Get metadata
                info = game_df.iloc[0]
                
                # Convert to time series
                timeseries = convert_game_to_timeseries(game_df)
                
                # Validate (should be 48 points)
                if len(timeseries) != 48:
                    errors += 1
                    continue
                
                # Extract Dejavu components
                pattern = timeseries[0:18]              # Minutes 0-17 (to 6:00 Q2)
                diff_at_pattern_end = timeseries[18]    # At 6:00 Q2
                diff_at_halftime = timeseries[24]       # At halftime
                diff_at_final = timeseries[47]          # At game end
                
                # Store everything
                all_games_data.append({
                    'season': season,
                    'game_id': game_url,
                    'date': info['Date'],
                    'home_team': info['HomeTeam'],
                    'away_team': info['AwayTeam'],
                    'winner': info['WinningTeam'],
                    'timeseries': timeseries,                    # Full 48 points
                    'pattern': pattern,                          # First 18 points
                    'diff_at_6min_Q2': diff_at_pattern_end,     # Differential at pattern end
                    'diff_at_halftime': diff_at_halftime,       # Our forecast target
                    'diff_at_final': diff_at_final,             # Final differential
                    'change_6min_to_HT': diff_at_halftime - diff_at_pattern_end,
                    'change_HT_to_final': diff_at_final - diff_at_halftime
                })
                
                games_processed += 1
                
            except Exception as e:
                errors += 1
                continue
        
        print(f"   ✅ Processed: {games_processed:,} games")
    
    elapsed = time.time() - start_time
    
    # Create master DataFrame
    print(f"\n{'='*80}")
    print("CREATING MASTER DATASET")
    print("="*80)
    
    games_df = pd.DataFrame(all_games_data)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total processing time:    {elapsed/60:.1f} minutes")
    print(f"   Total plays processed:    {total_plays:,}")
    print(f"   Total games:              {len(games_df):,}")
    print(f"   Errors:                   {errors}")
    
    print(f"\n   Breakdown by season:")
    for season in games_df['season'].unique():
        season_df = games_df[games_df['season'] == season]
        print(f"      {season}: {len(season_df):,} games")
    
    print(f"\n   Date range: {games_df['date'].min()} to {games_df['date'].max()}")
    print(f"   Teams: {games_df['home_team'].nunique()}")
    
    # Statistics on differentials
    print(f"\n📈 Differential Statistics:")
    print(f"   At 6:00 Q2 (pattern end):")
    print(f"      Mean: {games_df['diff_at_6min_Q2'].mean():+.2f}")
    print(f"      Std:  {games_df['diff_at_6min_Q2'].std():.2f}")
    print(f"      Range: [{games_df['diff_at_6min_Q2'].min():+.0f}, {games_df['diff_at_6min_Q2'].max():+.0f}]")
    
    print(f"\n   At halftime (forecast target):")
    print(f"      Mean: {games_df['diff_at_halftime'].mean():+.2f}")
    print(f"      Std:  {games_df['diff_at_halftime'].std():.2f}")
    print(f"      Range: [{games_df['diff_at_halftime'].min():+.0f}, {games_df['diff_at_halftime'].max():+.0f}]")
    
    print(f"\n   Change from 6:00 Q2 → halftime:")
    print(f"      Mean: {games_df['change_6min_to_HT'].mean():+.2f}")
    print(f"      Std:  {games_df['change_6min_to_HT'].std():.2f}")
    print(f"      Range: [{games_df['change_6min_to_HT'].min():+.0f}, {games_df['change_6min_to_HT'].max():+.0f}]")
    
    # Save datasets
    print(f"\n💾 Saving datasets...")
    
    # Save full dataset (with timeseries)
    games_df.to_pickle('complete_timeseries.pkl')
    print(f"   ✅ complete_timeseries.pkl ({len(games_df):,} games)")
    
    # Save pattern database (for Dejavu)
    # Extract just what we need for pattern matching
    pattern_db = []
    for idx, row in games_df.iterrows():
        pattern_db.append({
            'pattern': row['pattern'],
            'outcome': row['diff_at_halftime'],  # What we're predicting
            'diff_change': row['change_6min_to_HT'],
            'game_id': row['game_id'],
            'date': row['date'],
            'season': row['season'],
            'home_team': row['home_team'],
            'away_team': row['away_team']
        })
    
    import pickle
    with open('dejavu_pattern_database.pkl', 'wb') as f:
        pickle.dump(pattern_db, f)
    print(f"   ✅ dejavu_pattern_database.pkl ({len(pattern_db):,} patterns)")
    
    # Save summary statistics
    summary = {
        'total_games': len(games_df),
        'seasons': games_df['season'].unique().tolist(),
        'date_range': [games_df['date'].min(), games_df['date'].max()],
        'pattern_shape': pattern_db[0]['pattern'].shape,
        'processing_time_minutes': elapsed / 60,
        'diff_stats': {
            'mean_at_6min': float(games_df['diff_at_6min_Q2'].mean()),
            'std_at_6min': float(games_df['diff_at_6min_Q2'].std()),
            'mean_at_halftime': float(games_df['diff_at_halftime'].mean()),
            'std_at_halftime': float(games_df['diff_at_halftime'].std()),
            'mean_change': float(games_df['change_6min_to_HT'].mean()),
            'std_change': float(games_df['change_6min_to_HT'].std())
        }
    }
    
    import json
    with open('dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ dataset_summary.json")
    
    print(f"\n{'='*80}")
    print(f"✅ PROCESSING COMPLETE - {len(games_df):,} GAMES READY")
    print("="*80)
    print(f"\nDataset ready for Dejavu:")
    print(f"  • {len(games_df):,} game patterns")
    print(f"  • Pattern shape: {pattern_db[0]['pattern'].shape}")
    print(f"  • Outcome: halftime differential")
    print(f"\nNext step: Build Dejavu k=500 K-NN forecaster")
    
    return games_df, pattern_db


if __name__ == "__main__":
    games_df, pattern_db = process_all_seasons()

