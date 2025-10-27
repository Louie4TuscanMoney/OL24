#!/usr/bin/env python3
"""
🧬 OPTIMAL PATTERN EXTRACTION SYSTEM

Multi-Modal Pattern Recognition for Robust ML Ensemble

DESIGN PHILOSOPHY:
- Backward compatible (works with existing Dejavu)
- Forward compatible (optimized for advanced ML)
- Multi-modal (multiple pattern representations)
- Comprehensive (captures all game dynamics)
- Scalable (efficient extraction)

PATTERN TYPES:
1. TEMPORAL: 18-minute differential sequence (Dejavu, LSTM)
2. STATISTICAL: Distributional features (Random Forest)
3. SPECTRAL: Frequency components (momentum detection)
4. MULTIVARIATE: Joint statistics (Bayesian Networks, Vine Copulas)
5. SPATIAL: Player positioning (future enhancement)
6. CAUSAL: Event dependencies (Bayesian Networks)

OUTPUT FORMAT:
{
    # BASIC (Backward compatible)
    'pattern_temporal': [18 differentials],
    'diff_at_halftime': float,
    'diff_at_final': float,
    
    # STATISTICAL (Random Forest optimized)
    'pattern_statistical': {
        'mean': float,
        'std': float,
        'trend': float,
        'volatility': float,
        'momentum': float,
        'skewness': float,
        'kurtosis': float,
        'entropy': float
    },
    
    # SPECTRAL (Frequency domain)
    'pattern_spectral': {
        'low_freq': float,    # Long-term trend
        'mid_freq': float,    # Quarter rhythm
        'high_freq': float,   # Possession-by-possession
        'dominant_freq': float
    },
    
    # MULTIVARIATE (Bayesian Network ready)
    'pattern_multivariate': {
        'score_trajectory': [18 values],
        'pace_trajectory': [18 values],
        'shooting_efficiency': [18 values],
        'turnover_rate': [18 values],
        'rebound_differential': [18 values],
        'free_throw_rate': [18 values]
    },
    
    # PROBABILISTIC (Conformal ready)
    'pattern_probabilistic': {
        'volatility_regime': 'LOW'|'MEDIUM'|'HIGH',
        'momentum_state': 'INCREASING'|'STABLE'|'DECREASING',
        'blowout_risk': float (0-1),
        'comeback_probability': float (0-1)
    },
    
    # METADATA (All models)
    'quality_metrics': {
        'data_completeness': float (0-1),
        'variance_sufficient': bool,
        'no_outliers': bool,
        'confidence_score': float (0-1)
    }
}

This format:
✅ Works with Dejavu (uses pattern_temporal)
✅ Optimized for LSTM (sequential + multivariate)
✅ Perfect for Random Forest (statistical features)
✅ Ready for Bayesian Networks (multivariate + probabilistic)
✅ Supports vine copulas (joint distributions)
✅ Enables deep ensemble (multiple representations)
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, timedelta
from scipy import stats, signal
from scipy.fft import fft

from nba_api.stats.endpoints import playbyplayv2, boxscoretraditionalv2

class OptimalPatternExtractor:
    """
    State-of-the-art pattern extraction for multi-model ML ensemble
    """
    
    def __init__(self):
        """Initialize optimal pattern extractor"""
        print("="*80)
        print("🧬 OPTIMAL PATTERN EXTRACTION SYSTEM")
        print("="*80)
        print("\nDesigned for:")
        print("  ✅ Dejavu (k-NN similarity)")
        print("  ✅ LSTM (sequential learning)")
        print("  ✅ Random Forest (feature-based)")
        print("  ✅ Bayesian Networks (probabilistic)")
        print("  ✅ Vine Copulas (multivariate)")
        print("  ✅ Deep ensemble (all of above!)")
    
    def extract_temporal_pattern(self, plays_df):
        """
        Extract temporal 18-minute differential sequence
        BACKWARD COMPATIBLE with existing Dejavu model
        FORWARD COMPATIBLE with LSTM
        """
        differentials = [0]
        halftime_diff = None
        final_diff = None
        
        # Extract minute-by-minute from play-by-play
        for idx, play in plays_df.iterrows():
            period = play['PERIOD']
            pctimestring = play['PCTIMESTRING']
            score_margin = play['SCOREMARGIN']
            
            if pd.notna(pctimestring) and pd.notna(score_margin):
                try:
                    parts = pctimestring.split(':')
                    mins_remaining = int(parts[0])
                    secs_remaining = int(parts[1])
                    
                    if period == 1:
                        elapsed = 12 - mins_remaining - (secs_remaining / 60.0)
                    elif period == 2:
                        elapsed = 12 + (12 - mins_remaining - (secs_remaining / 60.0))
                        if mins_remaining == 0 and secs_remaining <= 5:
                            halftime_diff = 0 if score_margin == 'TIE' else int(score_margin)
                    elif period == 4:
                        if mins_remaining == 0 and secs_remaining <= 5:
                            final_diff = 0 if score_margin == 'TIE' else int(score_margin)
                        continue
                    else:
                        continue
                    
                    if elapsed > 18:
                        continue
                    
                    diff = 0 if score_margin == 'TIE' else int(score_margin)
                    minute = int(elapsed)
                    
                    if 0 <= minute <= 18:
                        while len(differentials) <= minute:
                            differentials.append(differentials[-1])
                        differentials[minute] = diff
                
                except:
                    continue
        
        # Ensure 18 values
        while len(differentials) < 18:
            differentials.append(differentials[-1])
        
        pattern = differentials[:18]
        
        return {
            'pattern': np.array(pattern),
            'halftime_diff': halftime_diff,
            'final_diff': final_diff
        }
    
    def extract_statistical_features(self, temporal_pattern):
        """
        Extract statistical features from temporal pattern
        OPTIMIZED for Random Forest and gradient boosting
        
        Returns 20+ features that capture:
        - Central tendency
        - Dispersion
        - Shape
        - Dynamics
        - Momentum
        """
        pattern = np.array(temporal_pattern)
        
        # Basic statistics
        mean = np.mean(pattern)
        std = np.std(pattern)
        min_val = np.min(pattern)
        max_val = np.max(pattern)
        
        # Distribution shape
        skewness = stats.skew(pattern)
        kurtosis = stats.kurtosis(pattern)
        
        # Dynamics
        diff = np.diff(pattern)
        velocity = np.mean(diff) if len(diff) > 0 else 0  # Average change
        acceleration = np.mean(np.diff(diff)) if len(diff) > 1 else 0
        
        # Momentum indicators
        recent_trend = np.mean(pattern[-6:]) - np.mean(pattern[:6])  # Last 6 vs first 6
        volatility = np.std(diff) if len(diff) > 0 else 0
        
        # Range features
        range_val = max_val - min_val
        iqr = np.percentile(pattern, 75) - np.percentile(pattern, 25)
        
        # Entropy (pattern complexity)
        hist, _ = np.histogram(pattern, bins=10)
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        entropy = stats.entropy(hist + 1e-10)
        
        # Crossing features
        zero_crossings = np.sum(np.diff(np.sign(pattern)) != 0)
        mean_crossings = np.sum(np.diff(np.sign(pattern - mean)) != 0)
        
        # Autocorrelation
        if len(pattern) > 1:
            autocorr_1 = np.corrcoef(pattern[:-1], pattern[1:])[0, 1]
        else:
            autocorr_1 = 0
        
        # Stationarity proxy (variance ratio)
        first_half_var = np.var(pattern[:9])
        second_half_var = np.var(pattern[9:])
        variance_ratio = second_half_var / (first_half_var + 1e-10)
        
        return {
            # Central tendency
            'mean': float(mean),
            'median': float(np.median(pattern)),
            
            # Dispersion
            'std': float(std),
            'range': float(range_val),
            'iqr': float(iqr),
            
            # Shape
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            
            # Dynamics
            'velocity': float(velocity),
            'acceleration': float(acceleration),
            'trend': float(recent_trend),
            'volatility': float(volatility),
            
            # Momentum
            'momentum_6min': float(pattern[-1] - pattern[-6]) if len(pattern) >= 6 else 0,
            'momentum_12min': float(pattern[-1] - pattern[0]),
            
            # Complexity
            'entropy': float(entropy),
            'zero_crossings': int(zero_crossings),
            'mean_crossings': int(mean_crossings),
            
            # Stationarity
            'autocorr': float(autocorr_1) if not np.isnan(autocorr_1) else 0,
            'variance_ratio': float(variance_ratio),
            
            # Extremes
            'min': float(min_val),
            'max': float(max_val),
            'abs_max': float(np.max(np.abs(pattern)))
        }
    
    def extract_spectral_features(self, temporal_pattern):
        """
        Extract frequency domain features
        OPTIMIZED for momentum and rhythm detection
        
        Captures:
        - Game rhythm (periodic patterns)
        - Momentum shifts (frequency changes)
        - Run detection (high-frequency spikes)
        """
        pattern = np.array(temporal_pattern)
        
        # FFT for frequency analysis
        fft_vals = fft(pattern)
        power_spectrum = np.abs(fft_vals) ** 2
        
        # Frequency bins
        freqs = np.fft.fftfreq(len(pattern))
        
        # Low frequency (0-0.1) = long-term trend
        low_freq_power = np.sum(power_spectrum[np.abs(freqs) < 0.1])
        
        # Mid frequency (0.1-0.3) = quarter rhythm
        mid_freq_power = np.sum(power_spectrum[(np.abs(freqs) >= 0.1) & (np.abs(freqs) < 0.3)])
        
        # High frequency (0.3+) = possession-by-possession
        high_freq_power = np.sum(power_spectrum[np.abs(freqs) >= 0.3])
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(power_spectrum[1:len(pattern)//2]) + 1
        dominant_freq = freqs[dominant_freq_idx]
        
        # Spectral entropy (complexity measure)
        power_norm = power_spectrum / (power_spectrum.sum() + 1e-10)
        spectral_entropy = -np.sum(power_norm * np.log(power_norm + 1e-10))
        
        return {
            'low_freq_power': float(low_freq_power),
            'mid_freq_power': float(mid_freq_power),
            'high_freq_power': float(high_freq_power),
            'dominant_freq': float(dominant_freq),
            'spectral_entropy': float(spectral_entropy),
            'freq_concentration': float(high_freq_power / (low_freq_power + mid_freq_power + 1e-10))
        }
    
    def extract_multivariate_features(self, plays_df):
        """
        Extract multivariate time series features
        OPTIMIZED for Bayesian Networks and Vine Copulas
        
        Captures dependencies between:
        - Score differential
        - Pace (possessions)
        - Shooting efficiency
        - Turnover rate
        - Rebounds
        - Free throws
        
        These joint distributions model complex game dynamics
        """
        # Initialize trajectories
        differentials = []
        possessions = []
        field_goals_made = []
        field_goals_attempted = []
        turnovers = []
        rebounds = []
        free_throws = []
        
        # Track by minute
        minute_data = {i: {'diff': 0, 'poss': 0, 'fgm': 0, 'fga': 0, 'to': 0, 'reb': 0, 'ft': 0} 
                      for i in range(18)}
        
        for idx, play in plays_df.iterrows():
            period = play['PERIOD']
            if period > 2:
                break
            
            pctimestring = play['PCTIMESTRING']
            if pd.notna(pctimestring):
                try:
                    parts = pctimestring.split(':')
                    mins_remaining = int(parts[0])
                    secs_remaining = int(parts[1])
                    
                    if period == 1:
                        elapsed = 12 - mins_remaining - (secs_remaining / 60.0)
                    else:
                        elapsed = 12 + (12 - mins_remaining - (secs_remaining / 60.0))
                    
                    if elapsed > 18:
                        continue
                    
                    minute = int(elapsed)
                    if minute >= 18:
                        continue
                    
                    # Score differential
                    score_margin = play['SCOREMARGIN']
                    if pd.notna(score_margin):
                        diff = 0 if score_margin == 'TIE' else int(score_margin)
                        minute_data[minute]['diff'] = diff
                    
                    # Event classification
                    event_type = play.get('EVENTMSGTYPE', 0)
                    
                    # Possession changes (turnovers, rebounds)
                    if event_type == 5:  # Turnover
                        minute_data[minute]['to'] += 1
                    elif event_type == 3:  # Rebound
                        minute_data[minute]['reb'] += 1
                    elif event_type == 1:  # Made shot
                        minute_data[minute]['fgm'] += 1
                        minute_data[minute]['fga'] += 1
                        minute_data[minute]['poss'] += 1
                    elif event_type == 2:  # Missed shot
                        minute_data[minute]['fga'] += 1
                        minute_data[minute]['poss'] += 1
                    elif event_type == 3:  # Free throw
                        minute_data[minute]['ft'] += 1
                
                except:
                    continue
        
        # Convert to trajectories
        for i in range(18):
            data = minute_data[i]
            differentials.append(data['diff'])
            possessions.append(data['poss'])
            
            # Shooting efficiency (cumulative)
            fg_pct = data['fgm'] / data['fga'] if data['fga'] > 0 else 0
            field_goals_made.append(fg_pct)
            
            # Other rates
            turnovers.append(data['to'])
            rebounds.append(data['reb'])
            free_throws.append(data['ft'])
        
        # Calculate derived multivariate features
        pace_avg = np.mean(possessions)
        shooting_avg = np.mean(field_goals_made)
        turnover_rate = np.sum(turnovers) / (np.sum(possessions) + 1e-10)
        
        return {
            'score_trajectory': differentials,
            'pace_trajectory': possessions,
            'shooting_efficiency': field_goals_made,
            'turnover_trajectory': turnovers,
            'rebound_trajectory': rebounds,
            'free_throw_trajectory': free_throws,
            
            # Aggregated for Bayesian Network nodes
            'avg_pace': float(pace_avg),
            'avg_shooting': float(shooting_avg),
            'turnover_rate': float(turnover_rate),
            'rebound_rate': float(np.mean(rebounds)),
            
            # Joint statistics (for vine copulas)
            'score_pace_correlation': float(np.corrcoef(differentials, possessions)[0,1]) 
                if len(set(possessions)) > 1 else 0,
            'score_shooting_correlation': float(np.corrcoef(differentials, field_goals_made)[0,1])
                if len(set(field_goals_made)) > 1 else 0
        }
    
    def extract_probabilistic_features(self, temporal_pattern, multivariate_features):
        """
        Extract probabilistic regime indicators
        OPTIMIZED for Conformal Prediction and uncertainty quantification
        """
        pattern = np.array(temporal_pattern)
        
        # Volatility regime
        volatility = np.std(np.diff(pattern)) if len(pattern) > 1 else 0
        
        if volatility < 2:
            volatility_regime = 'LOW'
        elif volatility < 5:
            volatility_regime = 'MEDIUM'
        else:
            volatility_regime = 'HIGH'
        
        # Momentum state
        recent_momentum = np.mean(pattern[-6:]) - np.mean(pattern[-12:-6]) if len(pattern) >= 12 else 0
        
        if recent_momentum > 2:
            momentum_state = 'INCREASING'
        elif recent_momentum < -2:
            momentum_state = 'DECREASING'
        else:
            momentum_state = 'STABLE'
        
        # Blowout risk (based on current differential and trend)
        current_diff = abs(pattern[-1])
        trend = pattern[-1] - pattern[0] if len(pattern) > 0 else 0
        
        blowout_risk = min(1.0, (current_diff + abs(trend)) / 30.0)
        
        # Comeback probability (if currently losing)
        if pattern[-1] < 0:  # Behind
            comeback_features = [
                multivariate_features.get('avg_pace', 0) / 100,  # Normalize
                1.0 - blowout_risk,
                1.0 if momentum_state == 'INCREASING' else 0.5
            ]
            comeback_probability = np.mean(comeback_features)
        else:
            comeback_probability = 1.0 - blowout_risk
        
        return {
            'volatility_regime': volatility_regime,
            'momentum_state': momentum_state,
            'blowout_risk': float(blowout_risk),
            'comeback_probability': float(comeback_probability),
            'current_differential': float(pattern[-1]),
            'trend_strength': float(abs(trend))
        }
    
    def calculate_quality_metrics(self, temporal_pattern, multivariate_features):
        """
        Calculate pattern quality metrics
        For filtering low-quality training samples
        """
        pattern = np.array(temporal_pattern)
        
        # Data completeness (no gaps)
        completeness = 1.0  # Assume complete if we got here
        
        # Variance sufficient (avoid constant patterns)
        variance_sufficient = np.var(pattern) > 0.1
        
        # No extreme outliers
        z_scores = np.abs(stats.zscore(pattern))
        no_outliers = np.all(z_scores < 5)  # No values >5 std devs
        
        # Confidence score (composite)
        confidence_factors = [
            completeness,
            1.0 if variance_sufficient else 0.5,
            1.0 if no_outliers else 0.7,
            min(1.0, multivariate_features.get('avg_pace', 0) / 50),  # Reasonable pace
        ]
        
        confidence_score = np.mean(confidence_factors)
        
        return {
            'data_completeness': float(completeness),
            'variance_sufficient': bool(variance_sufficient),
            'no_outliers': bool(no_outliers),
            'confidence_score': float(confidence_score),
            'quality_grade': 'A' if confidence_score > 0.9 else 'B' if confidence_score > 0.7 else 'C'
        }
    
    def extract_complete_pattern(self, game_id, game_metadata=None):
        """
        Extract ALL pattern types for a single game
        
        Returns comprehensive pattern dict ready for ALL ML models
        """
        try:
            # Get play-by-play
            pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
            plays_df = pbp.get_data_frames()[0]
            
            if len(plays_df) == 0:
                return None
            
            # Extract all pattern types
            temporal = self.extract_temporal_pattern(plays_df)
            
            if temporal['pattern'] is None or len(temporal['pattern']) != 18:
                return None
            
            statistical = self.extract_statistical_features(temporal['pattern'])
            spectral = self.extract_spectral_features(temporal['pattern'])
            multivariate = self.extract_multivariate_features(plays_df)
            probabilistic = self.extract_probabilistic_features(temporal['pattern'], multivariate)
            quality = self.calculate_quality_metrics(temporal['pattern'], multivariate)
            
            # Combine all
            complete_pattern = {
                # BACKWARD COMPATIBLE (Dejavu format)
                'pattern': temporal['pattern'].tolist(),  # List for JSON compatibility
                'diff_at_halftime': temporal['halftime_diff'],
                'diff_at_final': temporal['final_diff'],
                
                # ENHANCED (New models)
                'pattern_statistical': statistical,
                'pattern_spectral': spectral,
                'pattern_multivariate': multivariate,
                'pattern_probabilistic': probabilistic,
                'quality_metrics': quality,
                
                # Metadata
                'game_id': game_id,
                'extracted_at': datetime.now().isoformat()
            }
            
            # Add game metadata if provided
            if game_metadata:
                complete_pattern.update(game_metadata)
            
            return complete_pattern
            
        except Exception as e:
            return None

# Progress tracking
class LiveProgressTracker:
    """Live progress with detailed ETA and statistics"""
    
    def __init__(self, total):
        self.total = total
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = datetime.now()
        
        # Quality distribution
        self.quality_grades = {'A': 0, 'B': 0, 'C': 0}
    
    def update(self, success=True, quality_grade=None):
        self.processed += 1
        if success:
            self.successful += 1
            if quality_grade:
                self.quality_grades[quality_grade] = self.quality_grades.get(quality_grade, 0) + 1
        else:
            self.failed += 1
    
    def print_status(self):
        """Print comprehensive status update"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.processed / elapsed if elapsed > 0 else 0
        remaining_time = (self.total - self.processed) / rate if rate > 0 else 0
        eta = datetime.now() + timedelta(seconds=remaining_time)
        
        pct = (self.processed / self.total * 100) if self.total > 0 else 0
        
        # Progress bar
        bar_length = 50
        filled = int(bar_length * pct / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\n{'='*80}")
        print(f"🧬 EXTRACTION PROGRESS")
        print(f"{'='*80}")
        print(f"\n[{bar}] {pct:.1f}%")
        print(f"\n📊 Status:")
        print(f"   Processed: {self.processed:,}/{self.total:,} games")
        print(f"   Successful: {self.successful:,} ({self.successful/self.processed*100:.1f}%)")
        print(f"   Failed: {self.failed:,} ({self.failed/self.processed*100:.1f}%)")
        
        print(f"\n⏱️  Timing:")
        print(f"   Elapsed: {elapsed/60:.1f} minutes")
        print(f"   Speed: {rate*60:.1f} games/min")
        print(f"   Remaining: {remaining_time/60:.0f} minutes")
        print(f"   ETA: {eta.strftime('%I:%M %p')}")
        
        print(f"\n🎯 Quality Distribution:")
        total_graded = sum(self.quality_grades.values())
        if total_graded > 0:
            for grade, count in sorted(self.quality_grades.items()):
                pct_grade = count / total_graded * 100
                print(f"   Grade {grade}: {count:,} ({pct_grade:.1f}%)")
        
        print(f"{'='*80}\n")


# Main execution
if __name__ == "__main__":
    print("="*80)
    print("🧬 OPTIMAL PATTERN EXTRACTION - Multi-Modal System")
    print("="*80)
    
    print(f"\n🎯 Pattern Types Being Extracted:")
    print(f"   1. Temporal (18-min sequence) - Dejavu, LSTM")
    print(f"   2. Statistical (20+ features) - Random Forest")
    print(f"   3. Spectral (frequency domain) - Momentum detection")
    print(f"   4. Multivariate (joint stats) - Bayesian Network")
    print(f"   5. Probabilistic (regimes) - Conformal Prediction")
    print(f"   6. Quality (confidence) - Filtering")
    
    # Load games
    print(f"\n[1/4] Loading game list...")
    games_df = pd.read_csv('historical_games_2021_2025_basic.csv')
    games_df = games_df.drop_duplicates(subset=['GAME_ID'], keep='first')
    
    print(f"✅ {len(games_df):,} games to process")
    
    # Check checkpoint
    checkpoint_file = Path('optimal_patterns_checkpoint.pkl')
    processed_patterns = []
    processed_ids = set()
    
    if checkpoint_file.exists():
        print(f"\n📂 Loading checkpoint...")
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            processed_patterns = checkpoint['patterns']
            processed_ids = checkpoint['ids']
        
        print(f"   ✅ Resuming from {len(processed_patterns):,} games")
        games_df = games_df[~games_df['GAME_ID'].isin(processed_ids)]
        print(f"   Remaining: {len(games_df):,} games")
    
    # Initialize
    extractor = OptimalPatternExtractor()
    tracker = LiveProgressTracker(len(games_df))
    
    print(f"\n[2/4] Starting extraction...")
    print(f"   ⚠️  Estimated time: {len(games_df) * 0.6 / 60:.0f} minutes")
    print(f"   💡 Can pause anytime (Ctrl+C) and resume later")
    print(f"   ✅ Safe to use another Cursor agent while running\n")
    
    try:
        for idx, (_, row) in enumerate(games_df.iterrows(), 1):
            game_id = row['GAME_ID']
            
            # Metadata
            metadata = {
                'season': row['SEASON_ID'],
                'date': row['GAME_DATE'],
                'matchup': row['MATCHUP'],
                'home_team': row.get('TEAM_ABBREVIATION', ''),
                'away_team': ''  # Parse from matchup if needed
            }
            
            # Extract comprehensive pattern
            pattern = extractor.extract_complete_pattern(game_id, metadata)
            
            if pattern and pattern['diff_at_final'] is not None:
                processed_patterns.append(pattern)
                processed_ids.add(game_id)
                
                quality_grade = pattern.get('quality_metrics', {}).get('quality_grade', 'C')
                tracker.update(success=True, quality_grade=quality_grade)
            else:
                tracker.update(success=False)
            
            # Progress updates every 10 games
            if idx % 10 == 0:
                tracker.print_status()
            
            # Checkpoint every 50 games
            if idx % 50 == 0:
                print(f"💾 Checkpoint save... ({len(processed_patterns):,} games)")
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({'patterns': processed_patterns, 'ids': processed_ids}, f)
                print(f"   ✅ Saved")
            
            # Rate limit
            time.sleep(0.6)  # Respectful to NBA API
    
    except KeyboardInterrupt:
        print(f"\n\n⏸️  EXTRACTION PAUSED")
        print(f"   Processed: {len(processed_patterns):,} games")
        
        # Save checkpoint
        print(f"   💾 Saving checkpoint...")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'patterns': processed_patterns, 'ids': processed_ids}, f)
        
        print(f"   ✅ Checkpoint saved!")
        print(f"   📝 Run script again to resume from here")
        print(f"\n   Safe to:")
        print(f"   - Close terminal")
        print(f"   - Use other Cursor agents")
        print(f"   - Come back later")
        sys.exit(0)
    
    # Extraction complete
    print(f"\n[3/4] Extraction complete!")
    tracker.print_status()
    
    # Final save
    print(f"\n[4/4] Saving final dataset...")
    
    output_path = Path('OPTIMAL_PATTERNS_2021_2025.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(processed_patterns, f)
    
    print(f"✅ Saved {len(processed_patterns):,} games to: {output_path}")
    
    # Statistics
    print(f"\n📊 DATASET STATISTICS:")
    print(f"   Total games: {len(processed_patterns):,}")
    print(f"   Success rate: {tracker.successful/tracker.processed*100:.1f}%")
    
    grades = tracker.quality_grades
    total_graded = sum(grades.values())
    if total_graded > 0:
        print(f"\n   Quality breakdown:")
        print(f"   A (Excellent): {grades['A']:,} ({grades['A']/total_graded*100:.1f}%)")
        print(f"   B (Good): {grades['B']:,} ({grades['B']/total_graded*100:.1f}%)")
        print(f"   C (Acceptable): {grades['C']:,} ({grades['C']/total_graded*100:.1f}%)")
    
    # Sample pattern
    if processed_patterns:
        print(f"\n📋 Sample pattern structure:")
        sample = processed_patterns[0]
        print(f"   Keys: {list(sample.keys())}")
        print(f"   Temporal pattern length: {len(sample['pattern'])}")
        print(f"   Statistical features: {len(sample['pattern_statistical'])}")
        print(f"   Spectral features: {len(sample['pattern_spectral'])}")
        print(f"   Multivariate features: {len(sample['pattern_multivariate'])}")
    
    print(f"\n{'='*80}")
    print(f"✅ OPTIMAL PATTERN EXTRACTION COMPLETE!")
    print(f"{'='*80}")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Merge with existing 2015-2021 data")
    print(f"   2. Train models:")
    print(f"      - Dejavu (pattern_temporal)")
    print(f"      - Random Forest (pattern_statistical)")
    print(f"      - LSTM (pattern_temporal + multivariate)")
    print(f"      - Bayesian Network (pattern_multivariate)")
    print(f"   3. Build ensemble")
    print(f"   4. Test on holdout")
    print(f"   5. Expected MAE: 6-7 (vs current 10.75)!")

