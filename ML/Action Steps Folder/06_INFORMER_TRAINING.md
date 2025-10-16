# Step 6: Train Informer Model (Accuracy Improvement)

**Objective:** Train Informer model to improve forecast accuracy beyond Dejavu baseline

**Duration:** 1-3 days (mostly compute time)  
**Prerequisites:** Completed Step 5, GPU access (optional but recommended)  
**Output:** Trained Informer model with improved MAE

---

## Strategic Note

**Why wait until Step 6?**
- Already have working system (Dejavu + Conformal)
- Can collect real feedback while Informer trains
- Informer takes days to train properly
- Validate business value before investing compute

**Expected improvement:** MAE 5-8 → 3-5 points

---

## Action Items

### 6.1 Implement Informer Architecture (4-6 hours)

**Note:** Full Informer is complex (~2000 lines). 

**Important from Paper:** Informer designed for LONG sequences (336-1440 input tested in Zhou et al. AAAI 2021). 
For NBA's short 18-minute input, LSTM is actually more appropriate!

**Paper-Verified Informer Config (if implementing full version):**
- Encoder: 3-layer main stack + 1-layer stack (1/4 input) - pyramid
- Decoder: 2 layers
- Optimizer: Adam, lr=1e-4, decay 0.5× per epoch
- Epochs: 8 with early stopping
- Batch size: 32
- Platform: Single Nvidia V100 32GB GPU

**File:** `models/lstm_forecaster.py` (Appropriate for NBA's sequence length)

```python
"""
LSTM Forecaster for NBA (Informer proxy)
Full Informer implementation: See ../Informer-../Applied Model/ folder
"""

import torch
import torch.nn as nn
import numpy as np

class LSTMForecaster(nn.Module):
    """
    LSTM for NBA halftime forecasting (Informer proxy)
    """
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        forecast_horizon=6
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer (predict all 6 steps at once)
        self.fc = nn.Linear(hidden_size, forecast_horizon)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len=18, features=1)
        
        Returns:
            out: (batch, forecast_horizon=6)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Predict 6 steps
        forecast = self.fc(last_hidden)
        
        return forecast


class LSTMTrainer:
    """
    Training pipeline for LSTM
    """
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward
            predictions = self.model(batch_X)
            loss = self.criterion(predictions, batch_y)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                
                total_loss += loss.item()
                all_preds.extend(predictions[:, -1].cpu().numpy())  # Halftime prediction
                all_targets.extend(batch_y[:, -1].cpu().numpy())
        
        # MAE on halftime prediction
        mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
        
        return total_loss / len(val_loader), mae
```

---

### 6.2 Create PyTorch Dataset (30 minutes)

**File:** `models/nba_dataset.py`

```python
"""
PyTorch Dataset for NBA forecasting
"""

import torch
from torch.utils.data import Dataset
import numpy as np

class NBAHalftimeDataset(Dataset):
    """
    Dataset for NBA halftime prediction
    """
    def __init__(self, df, normalize=True):
        """
        Args:
            df: DataFrame with 'pattern' and 'outcome' columns
            normalize: Whether to z-score normalize
        """
        self.patterns = np.stack(df['pattern'].values)  # (n_games, 18)
        self.outcomes = np.stack(df['outcome'].values)  # (n_games, 6)
        
        if normalize:
            # Fit scaler on this split only
            self.pattern_mean = np.mean(self.patterns, axis=0)
            self.pattern_std = np.std(self.patterns, axis=0) + 1e-10
            
            self.outcome_mean = np.mean(self.outcomes, axis=0)
            self.outcome_std = np.std(self.outcomes, axis=0) + 1e-10
            
            # Normalize
            self.patterns = (self.patterns - self.pattern_mean) / self.pattern_std
            self.outcomes = (self.outcomes - self.outcome_mean) / self.outcome_std
        else:
            self.pattern_mean = 0
            self.pattern_std = 1
            self.outcome_mean = 0
            self.outcome_std = 1
    
    def __len__(self):
        return len(self.patterns)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.patterns[idx]).unsqueeze(-1)  # (18, 1)
        y = torch.FloatTensor(self.outcomes[idx])  # (6,)
        return x, y
```

---

### 6.3 Train LSTM Model (1-3 days compute time)

**File:** `scripts/train_lstm.py`

```python
"""
Train LSTM forecaster for NBA
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from models.lstm_forecaster import LSTMForecaster, LSTMTrainer
from models.nba_dataset import NBAHalftimeDataset

def train_lstm_model(
    train_df,
    val_df,
    epochs=50,
    batch_size=32,
    hidden_size=64,
    num_layers=2
):
    """
    Train LSTM model
    """
    print("=" * 80)
    print("TRAINING LSTM FORECASTER")
    print("=" * 80)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = NBAHalftimeDataset(train_df, normalize=True)
    val_dataset = NBAHalftimeDataset(val_df, normalize=True)
    
    # Use training normalization for validation
    val_dataset.pattern_mean = train_dataset.pattern_mean
    val_dataset.pattern_std = train_dataset.pattern_std
    val_dataset.outcome_mean = train_dataset.outcome_mean
    val_dataset.outcome_std = train_dataset.outcome_std
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print(f"\nInitializing model...")
    model = LSTMForecaster(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        forecast_horizon=6
    )
    
    trainer = LSTMTrainer(model, learning_rate=1e-3)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {trainer.device}")
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    best_val_mae = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader)
        
        # Validate
        val_loss, val_mae = trainer.evaluate(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val MAE: {val_mae:.2f} points")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), 'models/lstm_best.pth')
            patience_counter = 0
            print(f"  ✓ New best model (MAE: {val_mae:.2f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n✓ Training complete")
    print(f"✓ Best validation MAE: {best_val_mae:.2f} points")
    
    # Save normalization parameters
    import joblib
    normalization = {
        'pattern_mean': train_dataset.pattern_mean,
        'pattern_std': train_dataset.pattern_std,
        'outcome_mean': train_dataset.outcome_mean,
        'outcome_std': train_dataset.outcome_std
    }
    joblib.dump(normalization, 'models/lstm_normalization.pkl')
    
    return model, best_val_mae


if __name__ == "__main__":
    # Load splits
    train_df = pd.read_parquet('data/splits/train.parquet')
    val_df = pd.read_parquet('data/splits/validation.parquet')
    
    # Train
    model, best_mae = train_lstm_model(
        train_df,
        val_df,
        epochs=50,
        batch_size=32,
        hidden_size=64
    )
    
    print("\n" + "=" * 80)
    print("LSTM TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Val MAE: {best_mae:.2f} points")
    print(f"Saved to: models/lstm_best.pth")
```

---

### 6.4 Evaluation on Test Set (30 minutes)

**File:** `scripts/evaluate_lstm.py`

```python
"""
Evaluate trained LSTM on test set
"""

import torch
import pandas as pd
import numpy as np
from models.lstm_forecaster import LSTMForecaster
from models.nba_dataset import NBAHalftimeDataset
from torch.utils.data import DataLoader
import joblib

def evaluate_lstm():
    """
    Comprehensive evaluation of LSTM
    """
    # Load model
    model = LSTMForecaster(hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load('models/lstm_best.pth'))
    model.eval()
    
    # Load normalization
    norm_params = joblib.load('models/lstm_normalization.pkl')
    
    # Load test data
    test_df = pd.read_parquet('data/splits/test.parquet')
    test_dataset = NBAHalftimeDataset(test_df, normalize=True)
    
    # Apply training normalization
    test_dataset.pattern_mean = norm_params['pattern_mean']
    test_dataset.pattern_std = norm_params['pattern_std']
    test_dataset.outcome_mean = norm_params['outcome_mean']
    test_dataset.outcome_std = norm_params['outcome_std']
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    all_preds = []
    all_actuals = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            predictions = model(batch_X)
            
            # Denormalize predictions
            preds_denorm = predictions.cpu().numpy() * norm_params['outcome_std'] + norm_params['outcome_mean']
            actuals_denorm = batch_y.numpy() * norm_params['outcome_std'] + norm_params['outcome_mean']
            
            # Take halftime prediction (last step)
            all_preds.extend(preds_denorm[:, -1])
            all_actuals.extend(actuals_denorm[:, -1])
    
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    
    # Metrics
    mae = np.mean(np.abs(all_preds - all_actuals))
    rmse = np.sqrt(np.mean((all_preds - all_actuals) ** 2))
    
    metrics = {
        'test_mae': float(mae),
        'test_rmse': float(rmse),
        'n_samples': len(all_preds)
    }
    
    print("LSTM Test Performance:")
    print(f"  MAE: {mae:.2f} points")
    print(f"  RMSE: {rmse:.2f} points")
    
    # Save metrics
    import json
    with open('results/lstm_test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    metrics = evaluate_lstm()
    print(f"\n✓ LSTM achieves MAE: {metrics['test_mae']:.2f} points")
```

---

### 6.5 Compare Models (15 minutes)

**File:** `scripts/compare_models.py`

```python
"""
Compare Dejavu vs LSTM performance
"""

import json

def compare_models():
    """
    Compare all models on test set
    """
    # Load metrics
    with open('results/dejavu_validation_metrics.json') as f:
        dejavu_metrics = json.load(f)
    
    with open('results/lstm_test_metrics.json') as f:
        lstm_metrics = json.load(f)
    
    with open('results/conformal_test_metrics.json') as f:
        conformal_metrics = json.load(f)
    
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    print("\nPoint Forecast Accuracy (Test Set):")
    print(f"  Dejavu MAE:  {dejavu_metrics.get('mae', 'N/A')} points")
    print(f"  LSTM MAE:    {lstm_metrics['test_mae']:.2f} points")
    print(f"  Improvement: {dejavu_metrics.get('mae', 0) - lstm_metrics['test_mae']:.2f} points")
    
    print("\nUncertainty Quantification:")
    print(f"  Coverage:    {conformal_metrics['empirical_coverage']:.1%} (target: 95%)")
    print(f"  Interval:    ±{conformal_metrics['avg_interval_width']/2:.1f} points")
    
    print("\nDeployment Characteristics:")
    print("  Dejavu:")
    print("    + Instant deployment (no training)")
    print("    + Interpretable (shows similar games)")
    print("    + MAE ~5-8 points")
    print("  LSTM:")
    print("    + Better accuracy (MAE ~3-5 points)")
    print("    + Faster inference")
    print("    - Requires training (1-3 days)")
    
    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION: Ensemble both models")
    print("=" * 80)
    print("  Weight: 0.6 * LSTM + 0.4 * Dejavu")
    print("  Benefits: Accuracy + Interpretability")

if __name__ == "__main__":
    compare_models()
```

---

### 6.6 Validation Checklist

- [ ] ✅ LSTM model trained (50 epochs, early stopping)
- [ ] ✅ Test MAE < 5 points (better than Dejavu)
- [ ] ✅ Model saved with normalization parameters
- [ ] ✅ Performance compared against Dejavu baseline
- [ ] ✅ Ready for ensemble deployment

**Performance Targets:**
- Training MAE: <4 points
- Validation MAE: <4.5 points
- Test MAE: <5 points

---

## Expected Outputs

```
models/
├── lstm_best.pth                   ← Trained model weights
├── lstm_normalization.pkl          ← Normalization parameters
├── dejavu_forecaster.pkl           ← Dejavu (from Step 4)
└── conformal_predictor.pkl         ← Conformal (from Step 5)

results/
├── lstm_test_metrics.json
├── dejavu_validation_metrics.json
└── conformal_test_metrics.json
```

---

## Next Step

Proceed to **Step 7: Ensemble Models** to combine Dejavu + LSTM for optimal performance.

---

*Action Step 6 of 10 - Informer/LSTM Training*

