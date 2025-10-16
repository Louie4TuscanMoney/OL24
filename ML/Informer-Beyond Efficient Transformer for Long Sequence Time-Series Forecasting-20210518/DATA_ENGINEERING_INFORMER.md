# Data Engineering for Informer

**Production Data Pipeline Specifications**

**Date:** October 14, 2025  
**Model:** Informer Time Series Forecasting  
**Objective:** Optimal data preparation for Informer with SQL integration

---

## 📚 Complete Data Engineering Documentation

This folder contains complete data engineering specifications across multiple files:

| Resource | Location | Content |
|----------|----------|---------|
| **Comprehensive Guide** | `../DATA_ENGINEERING_SPEC.md` | Complete data engineering & feature engineering guide (1600+ lines) |
| **SQL Integration** | `Applied Model/sql_data_pipeline.py` | Production SQL data extraction with optimization |
| **Configuration** | `Applied Model/model_specs.py` | Data source & preprocessing configurations |
| **Quick Setup** | `Applied Model/QUICKSTART.py` | Interactive data pipeline setup |
| **This File** | `DATA_ENGINEERING_INFORMER.md` | Quick reference & Informer-specific notes |

---

## 🎯 Quick Reference: Informer-Specific Requirements

### Critical Data Requirements for Informer

1. **Temporal Features (ESSENTIAL)**
   ```python
   # Informer REQUIRES these features
   df['month'] = df['timestamp'].dt.month      # 1-12
   df['day'] = df['timestamp'].dt.day          # 1-31  
   df['weekday'] = df['timestamp'].dt.dayofweek  # 0-6
   df['hour'] = df['timestamp'].dt.hour        # 0-23
   ```

2. **Sequence Length Requirements**
   ```python
   # Input sequence (seq_len) must cover meaningful patterns
   seq_len = 96    # 4 days (hourly) or 1 day (15-min)
   seq_len = 168   # 1 week (hourly) ← Recommended for daily patterns
   seq_len = 720   # 1 month (hourly) ← For long-range dependencies
   ```

3. **Normalization (CRITICAL)**
   ```python
   # MUST fit scaler on TRAINING data only
   scaler = StandardScaler()
   train_data = scaler.fit_transform(train_data)  # Fit here
   val_data = scaler.transform(val_data)          # Transform only
   test_data = scaler.transform(test_data)        # Transform only
   ```

4. **Data Split (STRICT CHRONOLOGICAL)**
   ```python
   # NEVER shuffle time series data!
   train_end = int(n * 0.7)
   val_end = int(n * 0.8)
   
   train = df[:train_end]
   val = df[train_end:val_end]
   test = df[val_end:]
   ```

---

## 📊 Data Flow for Informer

```
SQL Database
    ↓
[Extract] (sql_data_pipeline.py)
    ↓
Raw Time Series Data
    ↓
[Clean] Missing values, outliers
    ↓
[Feature Engineering] Temporal features (month/day/hour)
    ↓
[Normalize] StandardScaler (fit on train only!)
    ↓
[Split] Train/Val/Test (chronological)
    ↓
[Create Datasets] Rolling windows (seq_len, pred_len)
    ↓
Informer Training
```

---

## 🔧 Implementation: Step-by-Step

### Step 1: SQL Data Extraction

**See:** `Applied Model/sql_data_pipeline.py`

```python
from sql_data_pipeline import SQLTimeSeriesPipeline
from model_specs import ConfigPresets

# 1. Create configuration
config = ConfigPresets.energy_forecasting()
config.data_source.host = "your-db.com"
config.data_schema.table_name = "energy_loads"

# 2. Extract data
pipeline = SQLTimeSeriesPipeline(config)
df = pipeline.run()  # Handles extraction, validation, cleaning

# 3. Save for faster iteration
pipeline.save_to_parquet("data/energy_data.parquet")
```

**Features:**
- ✅ Optimized SQL queries (filters applied at database)
- ✅ Connection pooling for performance
- ✅ Chunk-based loading for large datasets
- ✅ Automatic validation and quality checks

### Step 2: Data Cleaning

**See:** `../DATA_ENGINEERING_SPEC.md` → Section "Data Cleaning Pipeline"

```python
# Example: Clean extracted data
from data_engineering_pipeline import MissingValueHandler, OutlierDetector

# Handle missing values
missing_handler = MissingValueHandler()
df = missing_handler.fit_transform(df, time_col='timestamp', value_cols=value_columns)

# Detect and treat outliers
outlier_detector = OutlierDetector()
df = outlier_detector.fit_transform(df, value_cols=value_columns)
```

### Step 3: Feature Engineering

**Temporal Features (ESSENTIAL for Informer):**

```python
class InformerFeatureEngineer:
    """Extract temporal features for Informer"""
    
    def extract_temporal_features(self, df, time_col='timestamp'):
        """
        Informer expects these features in x_mark
        """
        df = df.copy()
        dt = pd.to_datetime(df[time_col]).dt
        
        # Core temporal features (REQUIRED)
        df['month'] = dt.month        # 1-12
        df['day'] = dt.day            # 1-31
        df['weekday'] = dt.dayofweek  # 0-6 (Monday=0)
        df['hour'] = dt.hour          # 0-23
        
        # Optional: For high-frequency data
        if '15min' in str(df[time_col].diff().mode()[0]):
            df['minute'] = dt.minute  # 0-59
        
        # Domain-specific (optional but helpful)
        df['is_weekend'] = (dt.dayofweek >= 5).astype(int)
        df['is_peak_hour'] = ((dt.hour >= 7) & (dt.hour <= 10) | 
                              (dt.hour >= 17) & (dt.hour <= 21)).astype(int)
        
        return df
```

**See Full Guide:** `../DATA_ENGINEERING_SPEC.md` → Section "Feature Engineering for Informer"

### Step 4: Normalization

**CRITICAL: Fit on Training Data ONLY**

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Split first (chronological!)
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.8)

train_df = df[:train_end].copy()
val_df = df[train_end:val_end].copy()
test_df = df[val_end:].copy()

# Normalize (fit on train only!)
scaler = StandardScaler()

value_cols = ['feature_1', 'feature_2', 'target']
train_df[value_cols] = scaler.fit_transform(train_df[value_cols])  # FIT here
val_df[value_cols] = scaler.transform(val_df[value_cols])          # TRANSFORM only
test_df[value_cols] = scaler.transform(test_df[value_cols])        # TRANSFORM only

# Save scaler for inference
import joblib
joblib.dump(scaler, 'scaler.pkl')
```

**See Full Guide:** `../DATA_ENGINEERING_SPEC.md` → Section "Normalization Strategies"

### Step 5: Create Informer Datasets

```python
class InformerDataset(torch.utils.data.Dataset):
    """
    Dataset for Informer model
    """
    def __init__(self, df, seq_len, label_len, pred_len, 
                 feature_cols, target_col, time_features):
        """
        Args:
            seq_len: Input sequence length (e.g., 96)
            label_len: Decoder start token length (e.g., 48)
            pred_len: Prediction horizon (e.g., 24)
            feature_cols: List of value column names
            target_col: Target column name
            time_features: ['month', 'day', 'weekday', 'hour']
        """
        self.data_x = df[feature_cols].values
        self.data_y = df[target_col].values
        self.data_stamp = df[time_features].values
        
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        # Encoder input
        s_begin = index
        s_end = s_begin + self.seq_len
        
        # Decoder input
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        
        seq_y_in = self.data_x[r_begin:r_end]
        seq_y_out = self.data_y[r_begin:r_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_in, seq_y_out, seq_y_mark
```

---

## 📋 Data Engineering Checklist

### Pre-Processing
- [ ] Extract data from SQL (use `sql_data_pipeline.py`)
- [ ] Validate timestamps (no duplicates, monotonic increasing)
- [ ] Handle missing values (forward fill, interpolation)
- [ ] Detect and treat outliers (IQR or Z-score methods)
- [ ] Check data quality (validation functions)

### Feature Engineering
- [ ] **Extract temporal features** (month, day, weekday, hour) ← CRITICAL
- [ ] Create cyclical encodings (optional: sin/cos for hour/day)
- [ ] Add domain-specific features (peak hours, weekends, holidays)
- [ ] Remove redundant features (high correlation)
- [ ] Validate feature distributions

### Normalization
- [ ] Choose normalization method (StandardScaler recommended)
- [ ] **Fit scaler on TRAINING data only** ← CRITICAL
- [ ] Transform validation and test sets
- [ ] Save scaler for inference
- [ ] Verify normalized distributions

### Data Splitting
- [ ] **Chronological split only** (NEVER shuffle!) ← CRITICAL
- [ ] Typical ratios: 70% train, 10% val, 20% test
- [ ] Validate no temporal overlap
- [ ] Ensure sufficient samples in each split
- [ ] Document split dates

### Dataset Creation
- [ ] Create PyTorch Dataset classes
- [ ] Configure sequence lengths (seq_len, label_len, pred_len)
- [ ] Include temporal features in x_mark
- [ ] Create DataLoaders with appropriate batch_size
- [ ] Test with sample batch

### Validation
- [ ] No data leakage (temporal, feature, normalization)
- [ ] No NaN or inf values
- [ ] Correct shapes for model input
- [ ] Distributions reasonable
- [ ] Coverage of temporal features complete

---

## ⚠️ Common Pitfalls & Solutions

### Pitfall 1: Missing Temporal Features
**Problem:** Informer performs poorly without hour/day/month features  
**Solution:** Always extract temporal features from timestamp

### Pitfall 2: Normalization Leakage
**Problem:** Fitting scaler on entire dataset  
**Solution:** Fit on training split only, transform val/test

### Pitfall 3: Data Shuffling
**Problem:** Shuffling time series breaks temporal order  
**Solution:** Always use chronological splits, never shuffle

### Pitfall 4: Wrong Sequence Lengths
**Problem:** seq_len too short to capture patterns  
**Solution:** Use seq_len ≥ 168 for daily patterns, ≥ 720 for weekly

### Pitfall 5: Ignoring Data Quality
**Problem:** Training on data with many outliers/missing values  
**Solution:** Always run data quality checks and cleaning

---

## 🔗 Complete Documentation Links

### Comprehensive Guides

1. **Full Data Engineering Guide**  
   Location: `../DATA_ENGINEERING_SPEC.md`  
   Content: 1600+ lines covering:
   - Data cleaning pipelines
   - Feature engineering for Informer & N-BEATS
   - Normalization strategies
   - Data validation
   - Complete implementation code

2. **SQL Data Pipeline**  
   Location: `Applied Model/sql_data_pipeline.py`  
   Content:
   - Database connection management
   - Optimized query building
   - Data extraction and validation
   - Quality checks

3. **Configuration System**  
   Location: `Applied Model/model_specs.py`  
   Content:
   - DataSourceConfig for SQL connections
   - DataTableSchema for table structure
   - DataPreprocessingConfig for cleaning
   - NormalizationConfig for scaling

4. **Quick Start Guide**  
   Location: `Applied Model/QUICKSTART.py`  
   Content: Interactive setup for data extraction

5. **Complete README**  
   Location: `Applied Model/README.md`  
   Content: Full usage examples and documentation

---

## 💡 Best Practices Summary

1. **Always Extract Temporal Features**
   ```python
   # Informer NEEDS these
   df['month'] = timestamp.dt.month
   df['day'] = timestamp.dt.day
   df['weekday'] = timestamp.dt.dayofweek
   df['hour'] = timestamp.dt.hour
   ```

2. **Normalize Per-Feature**
   ```python
   # Separate scaler per feature column
   scaler = StandardScaler()
   scaled_data = scaler.fit_transform(train_data)
   ```

3. **Never Leak Future Information**
   ```python
   # ✓ CORRECT: Forward fill only
   df['value'].fillna(method='ffill')
   
   # ✗ WRONG: Don't use backward fill
   # df['value'].fillna(method='bfill')  # Uses future!
   ```

4. **Validate Everything**
   ```python
   # Check for common issues
   assert df[time_col].is_monotonic_increasing
   assert not df[value_cols].isna().any()
   assert not df[value_cols].isin([np.inf, -np.inf]).any()
   ```

5. **Save Preprocessing State**
   ```python
   # Save for reproducibility and inference
   joblib.dump(scaler, 'scaler.pkl')
   config.to_json('config.json')
   df.to_parquet('preprocessed_data.parquet')
   ```

---

## 🚀 Quick Start Example

```python
# Complete data pipeline in 10 lines
from model_specs import ConfigPresets
from sql_data_pipeline import SQLTimeSeriesPipeline

# 1. Config
config = ConfigPresets.energy_forecasting()
config.data_source.host = "db.company.com"
config.data_schema.table_name = "loads"

# 2. Extract
pipeline = SQLTimeSeriesPipeline(config)
df = pipeline.run()

# 3. Done! Data is extracted, validated, and ready for training
```

For complete implementation, see:
- `Applied Model/QUICKSTART.py` - Interactive
- `Applied Model/deployment_template.py` - Full pipeline
- `../DATA_ENGINEERING_SPEC.md` - Complete guide

---

## 📊 Data Requirements by Use Case

### Energy Forecasting
- **Sequence Length:** 168 (1 week) to capture weekly patterns
- **Features:** Temperature, hour, day-of-week, is_holiday
- **Frequency:** Hourly
- **Normalization:** StandardScaler

### Financial Forecasting
- **Sequence Length:** 252 (1 year trading days)
- **Features:** Volume, market indicators, hour (for intraday)
- **Frequency:** Daily or minute
- **Normalization:** RobustScaler (handles outliers)

### Weather Forecasting
- **Sequence Length:** 144 (24 hours at 10-min intervals)
- **Features:** Multiple sensors, spatial features
- **Frequency:** 10-minute
- **Normalization:** StandardScaler per sensor

### Traffic Forecasting
- **Sequence Length:** 96 (24 hours at 15-min intervals)
- **Features:** Hour, day-of-week, is_rush_hour, events
- **Frequency:** 15-minute
- **Normalization:** StandardScaler

---

## 📝 Summary

**Data Engineering for Informer requires:**
- ✅ Temporal feature extraction (month/day/weekday/hour)
- ✅ Proper normalization (fit on train only)
- ✅ Chronological splits (never shuffle)
- ✅ Sufficient sequence lengths (≥96 timesteps)
- ✅ Data quality validation
- ✅ SQL integration for production

**Resources:**
- **Complete Guide:** `../DATA_ENGINEERING_SPEC.md` (1600+ lines)
- **SQL Pipeline:** `Applied Model/sql_data_pipeline.py`
- **Configuration:** `Applied Model/model_specs.py`
- **Quick Start:** `Applied Model/QUICKSTART.py`

**Result:** Production-ready data pipeline for Informer forecasting!

---

**Everything you need for optimal Informer data preparation** 📊

*Version 1.0.0 - October 14, 2025*

