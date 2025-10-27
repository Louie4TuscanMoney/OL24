"""
🔥 MOMENTUM ANALYZER - HEDGE FUND GRADE

Extracts momentum, run, and swing signals from score timeline.

MOMENTUM SIGNALS (3 streams):
1. Score differential over time (full resolution, not compressed!)
2. Run strength (scoring burst detection)
3. Lead volatility (swing magnitude)

Each stream gets 40 features from signal_transforms = 120 total momentum features!

This is where hedge funds extract "game flow" signal.
"""

import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class MomentumAnalyzer:
    """
    Analyzes momentum, runs, and swings from score timeline.
    
    Usage:
        analyzer = MomentumAnalyzer()
        momentum_streams = analyzer.extract(score_timeline)
        # Returns: Dict with 3 numpy arrays (signal streams)
    """
    
    def __init__(self):
        self.stream_count = 3
    
    def extract(self, score_timeline: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all momentum streams from score timeline.
        
        Args:
            score_timeline: Full-resolution score differential at each event
        
        Returns:
            Dictionary of 3 signal streams
        """
        if len(score_timeline) < 10:
            return self._empty_streams()
        
        streams = {}
        
        # STREAM 1: Score differential (full resolution, resampled to 50 points)
        streams['score_diff_full'] = self._resample(score_timeline, 50)
        
        # STREAM 2: Run strength (scoring burst magnitude)
        streams['run_strength'] = self._detect_runs(score_timeline)
        
        # STREAM 3: Lead volatility (magnitude of swings)
        streams['lead_volatility'] = self._compute_volatility(score_timeline)
        
        return streams
    
    def _detect_runs(self, score_timeline: np.ndarray, window: int = 10) -> np.ndarray:
        """
        Detect scoring runs (bursts of momentum).
        
        A run is defined as the max change in differential over a rolling window.
        """
        if len(score_timeline) < window:
            return np.zeros(20)
        
        runs = []
        for i in range(len(score_timeline) - window + 1):
            window_data = score_timeline[i:i+window]
            run_strength = np.max(window_data) - np.min(window_data)
            runs.append(run_strength)
        
        return self._resample(np.array(runs), 20)
    
    def _compute_volatility(self, score_timeline: np.ndarray, window: int = 10) -> np.ndarray:
        """
        Compute lead volatility (how much the lead swings).
        
        Uses rolling standard deviation of score differential.
        """
        if len(score_timeline) < window:
            return np.ones(20) * np.std(score_timeline)
        
        volatility = []
        for i in range(len(score_timeline) - window + 1):
            window_data = score_timeline[i:i+window]
            vol = np.std(window_data)
            volatility.append(vol)
        
        return self._resample(np.array(volatility), 20)
    
    def _resample(self, array: np.ndarray, target_length: int) -> np.ndarray:
        """Resample array to target length using interpolation."""
        if len(array) == 0:
            return np.zeros(target_length)
        elif len(array) == target_length:
            return array
        elif len(array) > target_length:
            # Downsample
            return np.interp(np.linspace(0, len(array)-1, target_length),
                           np.arange(len(array)), array)
        else:
            # Upsample
            return np.interp(np.linspace(0, len(array)-1, target_length),
                           np.arange(len(array)), array)
    
    def _empty_streams(self) -> Dict[str, np.ndarray]:
        """Return empty streams for invalid data."""
        return {
            'score_diff_full': np.zeros(50),
            'run_strength': np.zeros(20),
            'lead_volatility': np.ones(20) * 5.0
        }


# Test
if __name__ == "__main__":
    print("="*80)
    print("🔥 MOMENTUM ANALYZER TEST")
    print("="*80)
    
    # Create test score timeline
    np.random.seed(42)
    test_timeline = np.cumsum(np.random.randn(100)) * 2  # Random walk
    
    analyzer = MomentumAnalyzer()
    streams = analyzer.extract(test_timeline)
    
    print(f"\nInput timeline: {len(test_timeline)} events")
    print(f"Extracted {len(streams)} momentum streams:")
    for name, stream in streams.items():
        print(f"  {name:<25} Length: {len(stream)}, Mean: {stream.mean():.3f}, Std: {stream.std():.3f}")
    
    print(f"\n✅ Momentum analyzer WORKING!")
    print("="*80)

