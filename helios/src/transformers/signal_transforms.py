"""
🌊 SIGNAL TRANSFORMS - CORE HEDGE FUND FEATURE EXTRACTION

Integrates ALL research from 22 systems:
- FFT (Fast Fourier Transform)
- Wavelets (Multi-scale decomposition)
- Spectral Analysis (Centroid, Rolloff, Flux, Kurtosis, Skewness)
- Temporal Domain (Autocorrelation, Entropy, Energy, Zero-crossing)
- Peak Detection (Max/min peaks, ratios)
- Statistical Domain (Kurtosis, Skewness, RMS, MAD, ECDF)

This is the CORE of Project Helios - where raw signals become hedge fund features.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class SignalTransformer:
    """
    Comprehensive signal transformation suite.
    
    Takes a 1D time series signal and extracts 40+ features across:
    - Temporal domain
    - Frequency domain (FFT)
    - Wavelet domain
    - Peak detection
    - Statistical domain
    
    Usage:
        transformer = SignalTransformer()
        features = transformer.transform(signal, prefix='shots')
        # Returns: {'shots_fft_mean': 2.5, 'shots_entropy': 1.8, ...}
    """
    
    def __init__(self):
        self.feature_count = 0
    
    def transform(self, signal: np.ndarray, prefix: str = 'signal') -> Dict[str, float]:
        """
        Apply ALL transforms to a signal.
        
        Args:
            signal: 1D numpy array (time series)
            prefix: Feature name prefix (e.g., 'shots', 'possessions')
        
        Returns:
            Dictionary of features with prefix
        """
        if len(signal) < 3:
            return self._empty_features(prefix)
        
        # Clean signal
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        features = {}
        
        # Apply all transform families
        features.update(self._temporal_domain(signal, prefix))
        features.update(self._spectral_domain_fft(signal, prefix))
        features.update(self._wavelet_domain(signal, prefix))
        features.update(self._peak_detection(signal, prefix))
        features.update(self._statistical_domain(signal, prefix))
        
        return features
    
    def _temporal_domain(self, signal: np.ndarray, prefix: str) -> Dict[str, float]:
        """
        Temporal domain features (12 features).
        
        Based on research from ULTRA Signal Processing + MIT systems.
        """
        features = {}
        n = len(signal)
        
        # Autocorrelation (lag 1, 2, 3)
        for lag in [1, 2, 3]:
            if n > lag:
                try:
                    corr = np.corrcoef(signal[:n-lag], signal[lag:n])[0, 1]
                    features[f'{prefix}_autocorr_lag{lag}'] = 0 if np.isnan(corr) else corr
                except:
                    features[f'{prefix}_autocorr_lag{lag}'] = 0
            else:
                features[f'{prefix}_autocorr_lag{lag}'] = 0
        
        # Differences
        if n > 1:
            diffs = np.diff(signal)
            features[f'{prefix}_mean_abs_diff'] = np.mean(np.abs(diffs))
            features[f'{prefix}_mean_diff'] = np.mean(diffs)
            features[f'{prefix}_median_abs_diff'] = np.median(np.abs(diffs))
            features[f'{prefix}_median_diff'] = np.median(diffs)
            features[f'{prefix}_sum_abs_diff'] = np.sum(np.abs(diffs))
            
            # Distance (Euclidean path length)
            features[f'{prefix}_distance'] = np.sum(np.sqrt(1 + diffs**2))
        else:
            for feat in ['mean_abs_diff', 'mean_diff', 'median_abs_diff', 
                        'median_diff', 'sum_abs_diff', 'distance']:
                features[f'{prefix}_{feat}'] = 0
        
        # Total energy
        features[f'{prefix}_total_energy'] = np.sum(signal**2) / len(signal)
        
        # Entropy (from ULTRA system)
        try:
            hist, _ = np.histogram(signal, bins=min(10, len(signal)//2 + 1))
            hist = hist[hist > 0]
            if len(hist) > 0:
                prob = hist / hist.sum()
                features[f'{prefix}_entropy'] = -np.sum(prob * np.log2(prob))
            else:
                features[f'{prefix}_entropy'] = 0
        except:
            features[f'{prefix}_entropy'] = 0
        
        # Zero crossing rate
        if n > 1:
            signs = np.sign(signal)
            zero_crossings = np.sum(np.abs(np.diff(signs)) > 0) / (n - 1)
            features[f'{prefix}_zero_crossing_rate'] = zero_crossings
        else:
            features[f'{prefix}_zero_crossing_rate'] = 0
        
        # Slope (linear fit)
        try:
            x = np.arange(n)
            slope, _ = np.polyfit(x, signal, 1)
            features[f'{prefix}_slope'] = slope
        except:
            features[f'{prefix}_slope'] = 0
        
        return features
    
    def _spectral_domain_fft(self, signal: np.ndarray, prefix: str) -> Dict[str, float]:
        """
        Spectral domain features using FFT (10 features).
        
        Based on ULTRA Signal Processing research.
        """
        features = {}
        n = len(signal)
        
        try:
            # Compute FFT
            fft_vals = np.abs(fft(signal))
            fft_vals = fft_vals[:n//2 + 1]  # Take positive frequencies only
            
            if len(fft_vals) < 2:
                return self._empty_spectral_features(prefix)
            
            freqs = fftfreq(n, d=1.0)[:n//2 + 1]
            
            # FFT mean coefficient
            features[f'{prefix}_fft_mean'] = np.mean(fft_vals)
            
            # Fundamental frequency (dominant component)
            if len(fft_vals) > 0:
                features[f'{prefix}_fundamental_freq'] = freqs[np.argmax(fft_vals)]
            else:
                features[f'{prefix}_fundamental_freq'] = 0
            
            # Spectral energy
            features[f'{prefix}_spectral_energy'] = np.sum(fft_vals**2)
            
            # Spectral centroid
            if np.sum(fft_vals**2) > 0:
                features[f'{prefix}_spectral_centroid'] = np.sum(freqs * fft_vals**2) / np.sum(fft_vals**2)
            else:
                features[f'{prefix}_spectral_centroid'] = 0
            
            # Spectral spread
            centroid = features[f'{prefix}_spectral_centroid']
            if np.sum(fft_vals**2) > 0:
                features[f'{prefix}_spectral_spread'] = np.sqrt(
                    np.sum(((freqs - centroid)**2) * fft_vals**2) / np.sum(fft_vals**2)
                )
            else:
                features[f'{prefix}_spectral_spread'] = 0
            
            # Spectral rolloff (95% energy point)
            cumsum = np.cumsum(fft_vals**2)
            if cumsum[-1] > 0:
                rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
                features[f'{prefix}_spectral_rolloff'] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
            else:
                features[f'{prefix}_spectral_rolloff'] = 0
            
            # Spectral flux (would need previous frame, use variance as proxy)
            features[f'{prefix}_spectral_flux'] = np.var(fft_vals)
            
            # Max frequency component
            features[f'{prefix}_max_freq_component'] = np.max(fft_vals)
            
            # Spectral kurtosis
            features[f'{prefix}_spectral_kurtosis'] = kurtosis(fft_vals) if len(fft_vals) > 3 else 0
            
            # Spectral skewness
            features[f'{prefix}_spectral_skewness'] = skew(fft_vals) if len(fft_vals) > 2 else 0
            
        except Exception as e:
            return self._empty_spectral_features(prefix)
        
        return features
    
    def _wavelet_domain(self, signal: np.ndarray, prefix: str) -> Dict[str, float]:
        """
        Wavelet domain features (6 features).
        
        Simple multi-scale wavelet decomposition.
        """
        features = {}
        n = len(signal)
        
        try:
            # Scale 1 (full signal)
            features[f'{prefix}_wavelet_mean_s1'] = np.abs(np.mean(signal))
            features[f'{prefix}_wavelet_std_s1'] = np.std(signal)
            
            # Scale 2 (downsample by 2)
            if n >= 4:
                signal_s2 = signal[::2]
                features[f'{prefix}_wavelet_mean_s2'] = np.abs(np.mean(signal_s2))
                features[f'{prefix}_wavelet_std_s2'] = np.std(signal_s2)
            else:
                features[f'{prefix}_wavelet_mean_s2'] = np.abs(np.mean(signal))
                features[f'{prefix}_wavelet_std_s2'] = np.std(signal)
            
            # Wavelet energy
            features[f'{prefix}_wavelet_energy'] = np.sum(signal**2)
            
            # Wavelet variance
            features[f'{prefix}_wavelet_variance'] = np.var(signal)
            
        except:
            for feat in ['wavelet_mean_s1', 'wavelet_std_s1', 'wavelet_mean_s2', 
                        'wavelet_std_s2', 'wavelet_energy', 'wavelet_variance']:
                features[f'{prefix}_{feat}'] = 0
        
        return features
    
    def _peak_detection(self, signal: np.ndarray, prefix: str) -> Dict[str, float]:
        """
        Peak detection features (4 features).
        
        From Genetic Algorithm research.
        """
        features = {}
        n = len(signal)
        
        if n < 2:
            for feat in ['max_peaks', 'min_peaks', 'total_peaks', 'peak_ratio']:
                features[f'{prefix}_{feat}'] = 0
            return features
        
        diffs = np.diff(signal)
        
        # Maximum peaks (local maxima)
        max_peaks = 0
        for i in range(len(diffs) - 1):
            if diffs[i] > 0 and diffs[i+1] < 0:
                max_peaks += 1
        features[f'{prefix}_max_peaks'] = max_peaks
        
        # Minimum peaks (local minima)
        min_peaks = 0
        for i in range(len(diffs) - 1):
            if diffs[i] < 0 and diffs[i+1] > 0:
                min_peaks += 1
        features[f'{prefix}_min_peaks'] = min_peaks
        
        # Total peaks
        features[f'{prefix}_total_peaks'] = max_peaks + min_peaks
        
        # Peak ratio
        features[f'{prefix}_peak_ratio'] = max_peaks / (min_peaks + 1)
        
        return features
    
    def _statistical_domain(self, signal: np.ndarray, prefix: str) -> Dict[str, float]:
        """
        Statistical domain features (8 features).
        
        From Stanford/MIT research systems.
        """
        features = {}
        n = len(signal)
        
        try:
            # Basic stats
            mean = np.mean(signal)
            median = np.median(signal)
            
            # Kurtosis
            features[f'{prefix}_kurtosis'] = kurtosis(signal) if n > 3 else 0
            
            # Skewness
            features[f'{prefix}_skewness'] = skew(signal) if n > 2 else 0
            
            # Root mean square
            features[f'{prefix}_rms'] = np.sqrt(np.mean(signal**2))
            
            # Mean absolute deviation
            features[f'{prefix}_mad'] = np.mean(np.abs(signal - mean))
            
            # Median absolute deviation
            features[f'{prefix}_median_ad'] = np.median(np.abs(signal - median))
            
            # Variance
            features[f'{prefix}_variance'] = np.var(signal)
            
            # IQR
            features[f'{prefix}_iqr'] = np.percentile(signal, 75) - np.percentile(signal, 25)
            
            # Peak-to-peak distance
            features[f'{prefix}_peak_to_peak'] = np.max(signal) - np.min(signal)
            
        except:
            for feat in ['kurtosis', 'skewness', 'rms', 'mad', 'median_ad', 
                        'variance', 'iqr', 'peak_to_peak']:
                features[f'{prefix}_{feat}'] = 0
        
        return features
    
    def _empty_features(self, prefix: str) -> Dict[str, float]:
        """Return empty feature dict for invalid signals."""
        features = {}
        
        # Temporal (12)
        for lag in [1, 2, 3]:
            features[f'{prefix}_autocorr_lag{lag}'] = 0
        for feat in ['mean_abs_diff', 'mean_diff', 'median_abs_diff', 'median_diff',
                    'sum_abs_diff', 'distance', 'total_energy', 'entropy',
                    'zero_crossing_rate', 'slope']:
            features[f'{prefix}_{feat}'] = 0
        
        # Spectral (10)
        features.update(self._empty_spectral_features(prefix))
        
        # Wavelet (6)
        for feat in ['wavelet_mean_s1', 'wavelet_std_s1', 'wavelet_mean_s2',
                    'wavelet_std_s2', 'wavelet_energy', 'wavelet_variance']:
            features[f'{prefix}_{feat}'] = 0
        
        # Peaks (4)
        for feat in ['max_peaks', 'min_peaks', 'total_peaks', 'peak_ratio']:
            features[f'{prefix}_{feat}'] = 0
        
        # Statistical (8)
        for feat in ['kurtosis', 'skewness', 'rms', 'mad', 'median_ad',
                    'variance', 'iqr', 'peak_to_peak']:
            features[f'{prefix}_{feat}'] = 0
        
        return features
    
    def _empty_spectral_features(self, prefix: str) -> Dict[str, float]:
        """Return empty spectral features."""
        return {
            f'{prefix}_fft_mean': 0,
            f'{prefix}_fundamental_freq': 0,
            f'{prefix}_spectral_energy': 0,
            f'{prefix}_spectral_centroid': 0,
            f'{prefix}_spectral_spread': 0,
            f'{prefix}_spectral_rolloff': 0,
            f'{prefix}_spectral_flux': 0,
            f'{prefix}_max_freq_component': 0,
            f'{prefix}_spectral_kurtosis': 0,
            f'{prefix}_spectral_skewness': 0
        }
    
    def get_feature_count(self) -> int:
        """Return total number of features extracted per signal."""
        return 12 + 10 + 6 + 4 + 8  # Temporal + Spectral + Wavelet + Peaks + Statistical = 40


# Convenience function
def transform_signal(signal: np.ndarray, prefix: str = 'signal') -> Dict[str, float]:
    """
    Quick transform function.
    
    Usage:
        features = transform_signal(my_signal, prefix='shots')
    """
    transformer = SignalTransformer()
    return transformer.transform(signal, prefix)


if __name__ == "__main__":
    # Test the transformer
    print("="*80)
    print("🌊 SIGNAL TRANSFORMER TEST")
    print("="*80)
    
    # Create test signal
    t = np.linspace(0, 10, 100)
    test_signal = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 0.1, 100)
    
    # Transform
    transformer = SignalTransformer()
    features = transformer.transform(test_signal, prefix='test')
    
    print(f"\nInput signal: {len(test_signal)} points")
    print(f"Output features: {len(features)}")
    print(f"\nSample features:")
    for i, (name, value) in enumerate(list(features.items())[:10]):
        print(f"  {name:<35} {value:>10.4f}")
    
    print(f"\n✅ Transformer extracts {transformer.get_feature_count()} features per signal")
    print("="*80)

