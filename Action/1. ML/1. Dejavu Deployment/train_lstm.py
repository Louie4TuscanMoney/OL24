"""
Train LSTM Forecaster for NBA
Following: ML/Action Steps Folder/06_INFORMER_TRAINING.md

EXACT TRAINING SPECS (CORRECTED):
- Epochs: 50 (line 240)
- Batch size: 32 (line 241)
- Learning rate: 1e-3 (line 114)
- Optimizer: Adam (line 114)
- Loss: MSELoss (line 115)
- Gradient clipping: max_norm=1.0 (line 133)
- Early stopping: patience=5
- forecast_horizon: 7 (CORRECTED - must reach minute 24 = halftime)
- Expected MAE: ~4.0 points (MODELSYNERGY.md)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
import time

from lstm_model import LSTMForecaster

# PyTorch Dataset
class NBADataset(Dataset):
    """
    NBA halftime prediction dataset
    """
    def __init__(self, df, pattern_mean=None, pattern_std=None, 
                 outcome_mean=None, outcome_std=None):
        """
        Args:
            df: DataFrame with 'pattern' and 'outcome_6step'
            pattern_mean/std: For normalization (use training stats)
            outcome_mean/std: For normalization (use training stats)
        """
        # Stack arrays
        self.patterns = np.stack(df['pattern'].values).astype(np.float32)  # (n, 18)
        self.outcomes = np.stack(df['outcome_7step'].values).astype(np.float32)  # (n, 7)
        
        # Normalize
        if pattern_mean is None:
            # Fit on this split (training set)
            self.pattern_mean = np.mean(self.patterns, axis=0)
            self.pattern_std = np.std(self.patterns, axis=0) + 1e-10
            self.outcome_mean = np.mean(self.outcomes, axis=0)
            self.outcome_std = np.std(self.outcomes, axis=0) + 1e-10
        else:
            # Use provided stats (for val/test)
            self.pattern_mean = pattern_mean
            self.pattern_std = pattern_std
            self.outcome_mean = outcome_mean
            self.outcome_std = outcome_std
        
        # Apply normalization
        self.patterns = (self.patterns - self.pattern_mean) / self.pattern_std
        self.outcomes = (self.outcomes - self.outcome_mean) / self.outcome_std
    
    def __len__(self):
        return len(self.patterns)
    
    def __getitem__(self, idx):
        # LSTM expects (seq_len, features)
        x = torch.FloatTensor(self.patterns[idx]).unsqueeze(-1)  # (18, 1)
        y = torch.FloatTensor(self.outcomes[idx])  # (7,)
        return x, y


def train_lstm():
    """
    Train LSTM with exact research specifications
    """
    print("="*80)
    print("TRAINING LSTM FORECASTER")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    with open('splits/train.pkl', 'rb') as f:
        train_df = pickle.load(f)
    with open('splits/validation.pkl', 'rb') as f:
        val_df = pickle.load(f)
    
    print(f"✅ Training: {len(train_df)} games")
    print(f"✅ Validation: {len(val_df)} games")
    
    # Create datasets
    print("\nCreating datasets with normalization...")
    train_dataset = NBADataset(train_df)
    
    # Use training normalization for validation
    val_dataset = NBADataset(
        val_df,
        pattern_mean=train_dataset.pattern_mean,
        pattern_std=train_dataset.pattern_std,
        outcome_mean=train_dataset.outcome_mean,
        outcome_std=train_dataset.outcome_std
    )
    
    print(f"✅ Datasets created")
    print(f"   Pattern shape: {train_dataset.patterns.shape}")
    print(f"   Outcome shape: {train_dataset.outcomes.shape}")
    
    # Save normalization parameters (CRITICAL for inference)
    norm_params = {
        'pattern_mean': train_dataset.pattern_mean,
        'pattern_std': train_dataset.pattern_std,
        'outcome_mean': train_dataset.outcome_mean,
        'outcome_std': train_dataset.outcome_std
    }
    with open('lstm_normalization.pkl', 'wb') as f:
        pickle.dump(norm_params, f)
    print(f"   ✅ Normalization parameters saved")
    
    # Create data loaders (line 241: batch_size=32)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model (lines 52-64)
    print("\nInitializing LSTM model...")
    model = LSTMForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        forecast_horizon=7
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created")
    print(f"   Parameters: {total_params:,}")
    print(f"   Device: {device}")
    
    # Optimizer and loss (lines 114-115)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\nTraining for 50 epochs (with early stopping)...")
    print(f"="*80)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    epochs = 50
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (line 133: max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                mae = torch.mean(torch.abs(predictions - batch_y))
                
                val_loss += loss.item()
                val_mae += mae.item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'lstm_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered (patience={patience})")
                break
    
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: lstm_best.pth")
    print(f"\nNext: Evaluate on test set (expected MAE ~4.0 points)")


if __name__ == "__main__":
    train_lstm()

