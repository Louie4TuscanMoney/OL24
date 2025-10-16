"""
LSTM Forecaster for NBA Halftime Prediction
Following: ML/Action Steps Folder/06_INFORMER_TRAINING.md (lines 52-99)

EXACT SPECIFICATIONS (Research-Verified + CORRECTED):
- input_size = 1 (line 58)
- hidden_size = 64 (line 59)
- num_layers = 2 (line 60)
- dropout = 0.1 (line 61)
- forecast_horizon = 7 (CORRECTED: must reach minute 24 = halftime)
"""

import torch
import torch.nn as nn
import numpy as np

class LSTMForecaster(nn.Module):
    """
    LSTM for NBA halftime forecasting
    
    Architecture from 06_INFORMER_TRAINING.md lines 52-99
    """
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        forecast_horizon=7
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # LSTM layers (lines 71-77)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer - predict all 6 steps at once (line 80)
        self.fc = nn.Linear(hidden_size, forecast_horizon)
    
    def forward(self, x):
        """
        Forward pass (lines 82-99)
        
        Args:
            x: (batch, seq_len=18, features=1)
        
        Returns:
            out: (batch, forecast_horizon=6)
        """
        # LSTM forward (line 91)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state (line 94)
        last_hidden = lstm_out[:, -1, :]
        
        # Predict 7 steps (corrected to reach minute 24 = halftime)
        forecast = self.fc(last_hidden)
        
        return forecast


if __name__ == "__main__":
    print("="*80)
    print("LSTM MODEL (Research-Verified Architecture)")
    print("="*80)
    
    # Create model
    model = LSTMForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        forecast_horizon=7
    )
    
    print(f"\nModel Architecture:")
    print(f"  Input size:       {model.lstm.input_size}")
    print(f"  Hidden size:      {model.hidden_size}")
    print(f"  Num layers:       {model.num_layers}")
    print(f"  Forecast horizon: {model.forecast_horizon}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameters:")
    print(f"  Total:      {total_params:,}")
    print(f"  Trainable:  {trainable_params:,}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 4
    seq_len = 18
    test_input = torch.randn(batch_size, seq_len, 1)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"  Input shape:  {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  ✅ Forward pass successful")
    
    print(f"\n{'='*80}")
    print("✅ LSTM MODEL READY")
    print("="*80)
    print(f"\nNext: Build training script")

