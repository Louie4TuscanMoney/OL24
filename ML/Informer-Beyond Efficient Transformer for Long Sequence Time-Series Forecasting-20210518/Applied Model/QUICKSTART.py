"""
QUICKSTART GUIDE: Informer Model Implementation
================================================

This file demonstrates the simplest path from SQL data to predictions.
Copy and modify for your specific use case.

Author: Engineering Team
Date: October 14, 2025
"""

# ============================================================================
# STEP 0: IMPORTS
# ============================================================================

from model_specs import InformerConfig, ConfigPresets
from sql_data_pipeline import SQLTimeSeriesPipeline
import pandas as pd
import os

# Set environment variables (or use .env file)
os.environ["DB_HOST"] = "your-database-host.com"
os.environ["DB_PORT"] = "5432"
os.environ["DB_NAME"] = "timeseries_db"
os.environ["DB_USER"] = "your_username"
os.environ["DB_PASSWORD"] = "your_secure_password"


# ============================================================================
# OPTION 1: USE A PRESET (RECOMMENDED FOR BEGINNERS)
# ============================================================================

def quickstart_with_preset():
    """
    Fastest way to get started - use a preset configuration
    """
    print("=" * 80)
    print("QUICKSTART: Using Preset Configuration")
    print("=" * 80)
    
    # Choose a preset based on your domain:
    # - energy_forecasting()
    # - financial_forecasting()
    # - weather_forecasting()
    # - traffic_forecasting()
    
    config = ConfigPresets.energy_forecasting()
    
    # Minimal customization - just set your table name and columns
    config.data_schema.table_name = "your_time_series_table"
    config.data_schema.timestamp_column = "timestamp"
    config.data_schema.target_column = "value"
    config.data_schema.value_columns = [
        "feature_1",
        "feature_2",
        "feature_3",
        "value"
    ]
    
    # Set date range
    config.data_filter.start_date = "2023-01-01"
    config.data_filter.end_date = "2024-01-01"
    
    # Extract data
    print("\nExtracting data from database...")
    pipeline = SQLTimeSeriesPipeline(config)
    df = pipeline.run()
    
    print(f"\n✓ Success! Extracted {len(df)} rows")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Date range: {df[config.data_schema.timestamp_column].min()} to {df[config.data_schema.timestamp_column].max()}")
    
    # Save for later use
    pipeline.save_to_parquet("data/my_data.parquet")
    config.to_json("my_config.json")
    
    print("\n✓ Data saved to: data/my_data.parquet")
    print("✓ Config saved to: my_config.json")
    
    return df, config


# ============================================================================
# OPTION 2: CUSTOM CONFIGURATION FROM SCRATCH
# ============================================================================

def quickstart_from_scratch():
    """
    For advanced users who want full control
    """
    print("=" * 80)
    print("QUICKSTART: Custom Configuration")
    print("=" * 80)
    
    # Create blank configuration
    config = InformerConfig()
    
    # ========================================
    # 1. DATABASE CONNECTION
    # ========================================
    config.data_source.database_type = "postgresql"
    config.data_source.host = os.getenv("DB_HOST")
    config.data_source.port = int(os.getenv("DB_PORT"))
    config.data_source.database = os.getenv("DB_NAME")
    config.data_source.username = os.getenv("DB_USER")
    config.data_source.password = os.getenv("DB_PASSWORD")
    
    # ========================================
    # 2. TABLE SCHEMA
    # ========================================
    config.data_schema.table_name = "my_timeseries_table"
    config.data_schema.timestamp_column = "datetime"
    config.data_schema.target_column = "target_value"
    config.data_schema.value_columns = [
        "feature_a",
        "feature_b",
        "feature_c",
        "target_value"
    ]
    
    # ========================================
    # 3. DATA FILTERS
    # ========================================
    config.data_filter.start_date = "2023-01-01"
    config.data_filter.end_date = "2024-01-01"
    config.data_filter.exclude_nulls = True
    
    # ========================================
    # 4. MODEL ARCHITECTURE
    # ========================================
    config.architecture.enc_in = len(config.data_schema.value_columns)
    config.architecture.dec_in = len(config.data_schema.value_columns)
    config.architecture.c_out = 1
    
    # Sequence lengths (adjust based on your needs)
    config.architecture.seq_len = 96      # Lookback: 96 timesteps
    config.architecture.label_len = 48    # Start token: 48 timesteps
    config.architecture.pred_len = 24     # Forecast: 24 timesteps
    
    # Model size
    config.architecture.d_model = 512
    config.architecture.n_heads = 8
    config.architecture.e_layers = 2
    
    # ========================================
    # 5. TRAINING SETTINGS
    # ========================================
    config.training.batch_size = 32
    config.training.num_epochs = 10
    config.training.use_gpu = True
    
    # ========================================
    # 6. VALIDATE AND EXTRACT
    # ========================================
    config.validate()
    
    print("\nExtracting data from database...")
    pipeline = SQLTimeSeriesPipeline(config)
    df = pipeline.run()
    
    print(f"\n✓ Success! Extracted {len(df)} rows")
    
    # Save
    pipeline.save_to_parquet("data/custom_data.parquet")
    config.to_json("custom_config.json")
    
    return df, config


# ============================================================================
# OPTION 3: LOAD EXISTING PARQUET (SKIP SQL EXTRACTION)
# ============================================================================

def quickstart_from_parquet():
    """
    If you already have data extracted, load from Parquet
    """
    print("=" * 80)
    print("QUICKSTART: Load from Parquet")
    print("=" * 80)
    
    # Load configuration
    config = InformerConfig.from_json("my_config.json")
    
    # Load data
    df = pd.read_parquet("data/my_data.parquet")
    
    print(f"✓ Loaded {len(df)} rows from Parquet")
    print(f"✓ Columns: {list(df.columns)}")
    
    return df, config


# ============================================================================
# DATA QUALITY CHECK
# ============================================================================

def check_data_quality(df, config):
    """
    Quick data quality checks before training
    """
    print("\n" + "=" * 80)
    print("DATA QUALITY CHECK")
    print("=" * 80)
    
    timestamp_col = config.data_schema.timestamp_column
    target_col = config.data_schema.target_column
    value_cols = config.data_schema.value_columns
    
    # Basic info
    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df[timestamp_col].min()} to {df[timestamp_col].max()}")
    print(f"Total duration: {df[timestamp_col].max() - df[timestamp_col].min()}")
    
    # Missing values
    print("\nMissing values:")
    for col in value_cols:
        missing = df[col].isna().sum()
        missing_pct = (missing / len(df)) * 100
        print(f"  {col}: {missing} ({missing_pct:.2f}%)")
    
    # Target statistics
    print(f"\nTarget column '{target_col}' statistics:")
    print(df[target_col].describe())
    
    # Check for duplicates
    duplicates = df[timestamp_col].duplicated().sum()
    print(f"\nDuplicate timestamps: {duplicates}")
    
    # Check chronological order
    is_sorted = df[timestamp_col].is_monotonic_increasing
    print(f"Chronologically ordered: {'✓ Yes' if is_sorted else '✗ No'}")
    
    return True


# ============================================================================
# COMPLETE EXAMPLE WORKFLOW
# ============================================================================

def complete_workflow():
    """
    Complete end-to-end workflow
    """
    print("\n\n")
    print("#" * 80)
    print("# COMPLETE WORKFLOW: SQL DATA TO PREDICTIONS")
    print("#" * 80)
    
    # Step 1: Create configuration and extract data
    print("\n[STEP 1] Configuration and Data Extraction")
    df, config = quickstart_with_preset()
    
    # Step 2: Data quality check
    print("\n[STEP 2] Data Quality Check")
    check_data_quality(df, config)
    
    # Step 3: Data is ready - next steps would be:
    print("\n[STEP 3] Next Steps")
    print("✓ Data extracted and validated")
    print("✓ Ready for preprocessing (see DATA_ENGINEERING_SPEC.md)")
    print("✓ Ready for model training (implement based on IMPLEMENTATION_SPEC.md)")
    print("✓ Ready for deployment (see deployment_template.py)")
    
    print("\n" + "=" * 80)
    print("QUICKSTART COMPLETE!")
    print("=" * 80)
    print("\nYour data is ready at: data/my_data.parquet")
    print("Your config is at: my_config.json")
    print("\nNext: Implement model training using the specifications in IMPLEMENTATION_SPEC.md")


# ============================================================================
# USEFUL SNIPPETS
# ============================================================================

def snippet_change_database():
    """Switch between different databases"""
    config = InformerConfig()
    
    # PostgreSQL
    config.data_source.database_type = "postgresql"
    config.data_source.host = "postgres-host.com"
    config.data_source.port = 5432
    
    # MySQL
    config.data_source.database_type = "mysql"
    config.data_source.host = "mysql-host.com"
    config.data_source.port = 3306
    
    # BigQuery
    config.data_source.database_type = "bigquery"
    config.data_source.database = "my-project.my_dataset"


def snippet_multi_entity():
    """Configure for multi-entity forecasting"""
    config = ConfigPresets.energy_forecasting()
    
    # Set entity column
    config.data_schema.entity_column = "sensor_id"
    
    # Filter specific entities
    config.data_filter.entity_ids = ["SENSOR_001", "SENSOR_002", "SENSOR_003"]
    
    # Extract
    pipeline = SQLTimeSeriesPipeline(config)
    df = pipeline.run()
    
    # Process each entity separately
    for entity_id in df[config.data_schema.entity_column].unique():
        entity_df = df[df[config.data_schema.entity_column] == entity_id]
        print(f"{entity_id}: {len(entity_df)} rows")


def snippet_adjust_forecast_horizon():
    """Change forecast length and lookback window"""
    config = ConfigPresets.energy_forecasting()
    
    # Short-term forecast (1 hour ahead)
    config.architecture.seq_len = 24    # 24 hours lookback
    config.architecture.pred_len = 1    # 1 hour forecast
    
    # Medium-term forecast (1 day ahead)
    config.architecture.seq_len = 168   # 1 week lookback
    config.architecture.pred_len = 24   # 1 day forecast
    
    # Long-term forecast (1 week ahead)
    config.architecture.seq_len = 720   # 30 days lookback
    config.architecture.pred_len = 168  # 1 week forecast


def snippet_model_sizes():
    """Adjust model size based on data complexity"""
    config = InformerConfig()
    
    # Small model (fast, less accurate)
    config.architecture.d_model = 256
    config.architecture.d_ff = 1024
    config.architecture.n_heads = 4
    config.architecture.e_layers = 2
    
    # Medium model (balanced)
    config.architecture.d_model = 512
    config.architecture.d_ff = 2048
    config.architecture.n_heads = 8
    config.architecture.e_layers = 2
    
    # Large model (slow, more accurate)
    config.architecture.d_model = 768
    config.architecture.d_ff = 3072
    config.architecture.n_heads = 12
    config.architecture.e_layers = 4


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║           INFORMER MODEL - QUICKSTART GUIDE                        ║
    ║                                                                    ║
    ║  Choose an option:                                                 ║
    ║    1. Run complete workflow (recommended)                          ║
    ║    2. Use preset configuration                                     ║
    ║    3. Custom configuration                                         ║
    ║    4. Load from existing Parquet                                   ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        complete_workflow()
    elif choice == "2":
        df, config = quickstart_with_preset()
        check_data_quality(df, config)
    elif choice == "3":
        df, config = quickstart_from_scratch()
        check_data_quality(df, config)
    elif choice == "4":
        try:
            df, config = quickstart_from_parquet()
            check_data_quality(df, config)
        except FileNotFoundError:
            print("\n✗ Error: Parquet file not found. Run option 1, 2, or 3 first.")
    else:
        print("\n✗ Invalid choice. Please run again and select 1-4.")
    
    print("\n\n" + "=" * 80)
    print("For more information, see:")
    print("  - README.md - Full documentation")
    print("  - config_matrix.csv - All parameters explained")
    print("  - deployment_template.py - Complete pipeline")
    print("=" * 80)

