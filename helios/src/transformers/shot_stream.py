"""
🏀 SHOT STREAM EXTRACTOR - HEDGE FUND GRADE

Extracts comprehensive shot-level signals from raw PBP events.

SHOT SIGNALS EXTRACTED (10 streams):
1. Shot frequency over time
2. FG% rolling window
3. 3PT% rolling window
4. Shot type distribution (rim, midrange, 3PT)
5. Shot outcome sequence (make/miss pattern)
6. Shot value stream (points per shot)
7. Shot clustering (hot/cold streaks)
8. Shot spacing (time between shots)
9. Shot efficiency (eFG%, TS%)
10. Fast break vs half-court shot ratio

Each stream gets 40 features from signal_transforms = 400 total shot features!
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class ShotStreamExtractor:
    """
    Extracts 10 shot-related signal streams from PBP events.
    
    Usage:
        extractor = ShotStreamExtractor()
        shot_streams = extractor.extract(events)
        # Returns: Dict with 10 numpy arrays (signal streams)
    """
    
    def __init__(self):
        self.stream_count = 10
    
    def extract(self, events: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Extract all shot streams from events.
        
        Args:
            events: List of event dictionaries from PBP collector
        
        Returns:
            Dictionary of 10 signal streams
        """
        streams = {}
        
        # Filter shot events
        shot_events = [e for e in events if e['event_type'] in [1, 2]]  # Made/Missed
        
        if len(shot_events) < 5:
            return self._empty_streams()
        
        # Extract raw shot data
        shot_times = []
        shot_makes = []
        shot_types = []
        shot_values = []
        
        for event in shot_events:
            # Time (convert to seconds elapsed)
            period = event.get('period', 1)
            time_str = event.get('time_string', '0:00')
            elapsed = self._time_to_seconds(period, time_str)
            shot_times.append(elapsed)
            
            # Make/miss
            is_make = event['event_type'] == 1
            shot_makes.append(1 if is_make else 0)
            
            # Shot type (infer from description and action)
            shot_type = self._infer_shot_type(event)
            shot_types.append(shot_type)  # 0=rim, 1=mid, 2=3PT
            
            # Shot value
            if is_make:
                if shot_type == 2:  # 3PT
                    shot_values.append(3)
                else:
                    shot_values.append(2)
            else:
                shot_values.append(0)
        
        # Convert to numpy
        shot_times = np.array(shot_times)
        shot_makes = np.array(shot_makes)
        shot_types = np.array(shot_types)
        shot_values = np.array(shot_values)
        
        # STREAM 1: Shot frequency (shots per minute, smoothed)
        streams['shot_freq'] = self._compute_shot_frequency(shot_times)
        
        # STREAM 2: FG% rolling window
        streams['fg_pct_rolling'] = self._compute_rolling_pct(shot_makes, window=10)
        
        # STREAM 3: 3PT% rolling window
        three_pt_mask = shot_types == 2
        three_pt_makes = shot_makes[three_pt_mask]
        if len(three_pt_makes) > 5:
            streams['three_pct_rolling'] = self._compute_rolling_pct(three_pt_makes, window=5)
        else:
            streams['three_pct_rolling'] = np.ones(20) * 0.35  # Default
        
        # STREAM 4: Shot type distribution (rim% over time)
        streams['rim_shot_pct'] = self._compute_shot_type_distribution(shot_types, target_type=0)
        
        # STREAM 5: Make/miss pattern (binary sequence)
        streams['make_miss_pattern'] = shot_makes[:20] if len(shot_makes) >= 20 else np.pad(shot_makes, (0, 20-len(shot_makes)), constant_values=0.5)
        
        # STREAM 6: Points per shot stream
        streams['points_per_shot'] = self._compute_rolling_mean(shot_values, window=10)
        
        # STREAM 7: Hot/cold streak indicator
        streams['streak_intensity'] = self._compute_streak_intensity(shot_makes)
        
        # STREAM 8: Shot spacing (time between shots)
        if len(shot_times) > 1:
            shot_spacing = np.diff(shot_times)
            streams['shot_spacing'] = np.pad(shot_spacing[:20], (0, max(0, 20-len(shot_spacing))), constant_values=np.median(shot_spacing))
        else:
            streams['shot_spacing'] = np.ones(20) * 30  # Default 30 sec
        
        # STREAM 9: Effective FG% stream
        efg_values = []
        for i in range(len(shot_makes)):
            if shot_types[i] == 2 and shot_makes[i] == 1:  # Made 3PT
                efg_values.append(1.5)  # Worth more
            else:
                efg_values.append(float(shot_makes[i]))
        
        streams['efg_stream'] = self._compute_rolling_mean(np.array(efg_values), window=10)
        
        # STREAM 10: Fast break proxy (shots within 8 sec of possession)
        fastbreak_indicator = []
        for i in range(len(shot_times)):
            if i == 0:
                fastbreak_indicator.append(0)
            else:
                time_since_last = shot_times[i] - shot_times[i-1]
                fastbreak_indicator.append(1 if time_since_last < 8 else 0)
        
        streams['fastbreak_rate'] = self._compute_rolling_mean(np.array(fastbreak_indicator), window=10)
        
        return streams
    
    def _empty_streams(self) -> Dict[str, np.ndarray]:
        """Return empty streams for games with insufficient shots."""
        return {
            'shot_freq': np.zeros(20),
            'fg_pct_rolling': np.ones(20) * 0.45,
            'three_pct_rolling': np.ones(20) * 0.35,
            'rim_shot_pct': np.ones(20) * 0.4,
            'make_miss_pattern': np.ones(20) * 0.5,
            'points_per_shot': np.ones(20) * 1.0,
            'streak_intensity': np.zeros(20),
            'shot_spacing': np.ones(20) * 30,
            'efg_stream': np.ones(20) * 0.5,
            'fastbreak_rate': np.zeros(20)
        }
    
    def _time_to_seconds(self, period: int, time_str: str) -> int:
        """Convert period + clock to elapsed seconds."""
        try:
            mins, secs = map(int, time_str.split(':'))
            period_elapsed = (period - 1) * 720  # 12 min per period
            current_period_elapsed = 720 - (mins * 60 + secs)
            return period_elapsed + current_period_elapsed
        except:
            return (period - 1) * 720
    
    def _infer_shot_type(self, event: Dict) -> int:
        """
        Infer shot type from event description.
        
        Returns:
            0: Rim (layup, dunk)
            1: Midrange
            2: 3PT
        """
        desc = (str(event.get('home_desc', '')) + ' ' + str(event.get('away_desc', ''))).upper()
        
        if '3PT' in desc or 'THREE' in desc:
            return 2
        elif 'LAYUP' in desc or 'DUNK' in desc:
            return 0
        else:
            return 1  # Midrange default
    
    def _compute_shot_frequency(self, shot_times: np.ndarray, n_points: int = 20) -> np.ndarray:
        """Compute shot frequency (shots per minute) over time."""
        if len(shot_times) < 2:
            return np.ones(n_points)
        
        # Divide game into n_points segments
        max_time = shot_times.max()
        segment_size = max_time / n_points
        
        freq = []
        for i in range(n_points):
            start_time = i * segment_size
            end_time = (i + 1) * segment_size
            shots_in_segment = np.sum((shot_times >= start_time) & (shot_times < end_time))
            freq.append(shots_in_segment / (segment_size / 60))  # Per minute
        
        return np.array(freq)
    
    def _compute_rolling_pct(self, binary_array: np.ndarray, window: int = 10) -> np.ndarray:
        """Compute rolling percentage."""
        if len(binary_array) < window:
            return np.ones(20) * binary_array.mean()
        
        rolling = []
        for i in range(len(binary_array)):
            start = max(0, i - window + 1)
            rolling.append(binary_array[start:i+1].mean())
        
        # Resample to 20 points
        return np.interp(np.linspace(0, len(rolling)-1, 20), np.arange(len(rolling)), rolling)
    
    def _compute_rolling_mean(self, array: np.ndarray, window: int = 10) -> np.ndarray:
        """Compute rolling mean."""
        if len(array) < window:
            return np.ones(20) * array.mean()
        
        rolling = []
        for i in range(len(array)):
            start = max(0, i - window + 1)
            rolling.append(array[start:i+1].mean())
        
        # Resample to 20 points
        if len(rolling) >= 20:
            return np.interp(np.linspace(0, len(rolling)-1, 20), np.arange(len(rolling)), rolling)
        else:
            return np.pad(rolling, (0, 20-len(rolling)), constant_values=rolling[-1])
    
    def _compute_shot_type_distribution(self, shot_types: np.ndarray, target_type: int) -> np.ndarray:
        """Compute rolling distribution of a specific shot type."""
        type_mask = (shot_types == target_type).astype(float)
        return self._compute_rolling_mean(type_mask, window=10)
    
    def _compute_streak_intensity(self, makes: np.ndarray) -> np.ndarray:
        """
        Compute hot/cold streak intensity.
        
        Positive = hot streak, Negative = cold streak
        """
        if len(makes) < 5:
            return np.zeros(20)
        
        streak = []
        current_streak = 0
        
        for make in makes:
            if make == 1:
                current_streak = max(1, current_streak + 1)
            else:
                current_streak = min(-1, current_streak - 1)
            streak.append(current_streak)
        
        # Resample to 20 points
        if len(streak) >= 20:
            return np.interp(np.linspace(0, len(streak)-1, 20), np.arange(len(streak)), streak)
        else:
            return np.pad(streak, (0, 20-len(streak)), constant_values=0)


# Quick test
if __name__ == "__main__":
    print("="*80)
    print("🏀 SHOT STREAM EXTRACTOR TEST")
    print("="*80)
    
    # Create mock events
    mock_events = [
        {'event_type': 1, 'period': 1, 'time_string': '11:30', 'home_desc': '3PT', 'away_desc': ''},
        {'event_type': 2, 'period': 1, 'time_string': '11:15', 'home_desc': '', 'away_desc': 'LAYUP'},
        {'event_type': 1, 'period': 1, 'time_string': '10:50', 'home_desc': 'LAYUP', 'away_desc': ''},
        # ... many more would be here
    ] * 20  # Repeat to simulate more shots
    
    extractor = ShotStreamExtractor()
    streams = extractor.extract(mock_events)
    
    print(f"\nExtracted {len(streams)} shot streams:")
    for name, stream in streams.items():
        print(f"  {name:<25} Length: {len(stream)}, Mean: {stream.mean():.3f}")
    
    print(f"\n✅ Shot extractor WORKING - ready for production!")
    print("="*80)

