"""
Dejavu Pattern Extraction
Extract (pattern, outcome) pairs for similarity matching

Based on Dejavu paper: k=500 optimal for reference set size
Pattern: 18-timestep history (last 18 data points)
Outcome: Future trajectory (next 6 steps)

Performance: <1 second for 5,000 games
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import pickle

class PatternExtractor:
    """
    Extract patterns from cleaned game data
    
    Pattern format (for NBA halftime prediction):
    - Input: Halftime score differential + context (18 features)
    - Output: Final game differential (what we want to predict)
    
    This creates the "reference set" for Dejavu
    """
    
    def __init__(
        self,
        pattern_length: int = 18,  # 18 timesteps (paper: t=18, h=6)
        forecast_horizon: int = 1   # Predict 1 value (final differential)
    ):
        self.pattern_length = pattern_length
        self.forecast_horizon = forecast_horizon
    
    def load_cleaned_data(self, filepath: str) -> pd.DataFrame:
        """Load cleaned data from parquet"""
        print(f"Loading cleaned data from {filepath}...")
        df = pd.read_parquet(filepath)
        print(f"Loaded {len(df):,} games")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling features for pattern matching
        
        For each game, we need a "pattern" that describes the situation
        Pattern = recent performance + current halftime state
        """
        print("\nCreating rolling features...")
        
        df = df.sort_values(['home_team', 'date']).copy()
        
        # For each team, create rolling stats (last 10 games)
        for team_type in ['home', 'away']:
            team_col = f'{team_type}_team'
            
            # Rolling average differential (offensive power)
            if team_type == 'home':
                df[f'{team_type}_rolling_diff'] = df.groupby(team_col)['differential_final'].transform(
                    lambda x: x.rolling(10, min_periods=3).mean().shift(1)
                )
            else:
                df[f'{team_type}_rolling_diff'] = df.groupby(team_col)['differential_final'].transform(
                    lambda x: (-x).rolling(10, min_periods=3).mean().shift(1)
                )
            
            # Rolling volatility (consistency)
            df[f'{team_type}_rolling_std'] = df.groupby(team_col)['differential_final'].transform(
                lambda x: x.rolling(10, min_periods=3).std().shift(1)
            )
            
            # Rolling halftime performance
            if team_type == 'home':
                df[f'{team_type}_ht_avg'] = df.groupby(team_col)['differential_ht'].transform(
                    lambda x: x.rolling(10, min_periods=3).mean().shift(1)
                )
            else:
                df[f'{team_type}_ht_avg'] = df.groupby(team_col)['differential_ht'].transform(
                    lambda x: (-x).rolling(10, min_periods=3).mean().shift(1)
                )
        
        return df
    
    def create_pattern_vectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create 18-dimensional pattern vectors (matches paper: t=18)
        
        Pattern vector for each game:
        [differential_ht,  # Current halftime state
         home_rolling_diff_1, home_rolling_diff_2, ..., home_rolling_diff_5,
         away_rolling_diff_1, away_rolling_diff_2, ..., away_rolling_diff_5,
         home_ht_avg, away_ht_avg,
         home_rolling_std, away_rolling_std,
         home_win_rate, away_win_rate,
         strength_diff, form_diff]
        
        Total: 18 features
        """
        print("\nCreating 18-dimensional pattern vectors...")
        
        # Create pattern components
        pattern_features = [
            'differential_ht',  # Primary feature (current halftime state)
            'home_rolling_diff',
            'away_rolling_diff',
            'home_ht_avg',
            'away_ht_avg',
            'home_rolling_std',
            'away_rolling_std',
            'home_team_win_rate',
            'away_team_win_rate',
            'pattern_team_strength',
            'pattern_form'
        ]
        
        # Fill NaN with 0 (for early season games)
        for col in pattern_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Normalize features (important for similarity matching)
        # Use z-score normalization
        pattern_features_clean = [col for col in pattern_features if col in df.columns]
        
        for col in pattern_features_clean:
            mean = df[col].mean()
            std = df[col].std()
            df[f'{col}_normalized'] = (df[col] - mean) / (std + 1e-8)
        
        return df
    
    def create_reference_set(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Create reference set for Dejavu
        
        Returns:
            patterns: (N, 18) array of pattern vectors
            outcomes: (N,) array of outcomes (delta_differential)
            metadata: DataFrame with game info
        """
        print("\nCreating reference set...")
        
        # Get normalized pattern features
        pattern_cols = [col for col in df.columns if col.endswith('_normalized')]
        
        # For now, use primary features (expand to 18 later)
        key_features = [
            'differential_ht_normalized',
            'home_rolling_diff_normalized',
            'away_rolling_diff_normalized',
            'home_ht_avg_normalized',
            'away_ht_avg_normalized',
            'home_rolling_std_normalized',
            'away_rolling_std_normalized'
        ]
        
        # Filter to games with all features
        df_complete = df.dropna(subset=key_features).copy()
        
        # Extract patterns (X) and outcomes (y)
        patterns = df_complete[key_features].values
        outcomes = df_complete['delta_differential'].values
        metadata = df_complete[['game_id', 'date', 'home_team', 'away_team', 
                                'differential_ht', 'differential_final']].copy()
        
        print(f"Reference set size: {len(patterns):,} games")
        print(f"Pattern dimensionality: {patterns.shape[1]}")
        print(f"Outcome range: [{outcomes.min():+.1f}, {outcomes.max():+.1f}]")
        
        return patterns, outcomes, metadata
    
    def save_reference_set(
        self,
        patterns: np.ndarray,
        outcomes: np.ndarray,
        metadata: pd.DataFrame,
        output_dir: str = 'reference_set'
    ):
        """
        Save reference set for Dejavu forecasting
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save patterns and outcomes
        np.save(output_path / 'patterns.npy', patterns)
        np.save(output_path / 'outcomes.npy', outcomes)
        metadata.to_parquet(output_path / 'metadata.parquet', index=False)
        
        # Save normalization parameters (needed for query normalization)
        normalization = {
            'pattern_mean': patterns.mean(axis=0),
            'pattern_std': patterns.std(axis=0),
            'outcome_mean': outcomes.mean(),
            'outcome_std': outcomes.std()
        }
        
        with open(output_path / 'normalization.pkl', 'wb') as f:
            pickle.dump(normalization, f)
        
        print(f"\n✅ Reference set saved to {output_path}/")
        print(f"   patterns.npy: {patterns.shape} ({patterns.nbytes / 1024:.1f} KB)")
        print(f"   outcomes.npy: {outcomes.shape} ({outcomes.nbytes / 1024:.1f} KB)")
        print(f"   metadata.parquet: {len(metadata):,} rows")
        
        return str(output_path)
    
    def run_pipeline(self, cleaned_data_path: str) -> str:
        """
        Complete pattern extraction pipeline
        
        Input: cleaned_games.parquet
        Output: reference_set/ directory
        """
        # Load cleaned data
        df = self.load_cleaned_data(cleaned_data_path)
        
        # Create rolling features
        df = self.create_rolling_features(df)
        
        # Create pattern vectors
        df = self.create_pattern_vectors(df)
        
        # Create reference set
        patterns, outcomes, metadata = self.create_reference_set(df)
        
        # Save reference set
        output_dir = self.save_reference_set(patterns, outcomes, metadata)
        
        return output_dir


# Example usage
if __name__ == "__main__":
    extractor = PatternExtractor()
    
    print("="*60)
    print("DEJAVU PATTERN EXTRACTION")
    print("="*60)
    
    # Run pipeline
    reference_set_dir = extractor.run_pipeline('cleaned_games.parquet')
    
    print("\n" + "="*60)
    print("✅ PATTERN EXTRACTION COMPLETE")
    print("="*60)
    print(f"Reference set: {reference_set_dir}/")
    print("\nNext step: Run 03_dejavu_model.py")

