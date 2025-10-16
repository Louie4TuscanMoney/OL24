"""
Production Deployment Template for Informer Model
==================================================

Complete deployment pipeline from SQL data to production API.

This template demonstrates the end-to-end workflow:
1. Data extraction from SQL
2. Preprocessing and feature engineering
3. Model training
4. Model evaluation
5. Model deployment
6. API serving

Author: Engineering Team
Date: October 14, 2025
Version: 1.0.0
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import logging

# Import our modules
from model_specs import InformerConfig, ConfigPresets
from sql_data_pipeline import SQLTimeSeriesPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# STEP 1: CONFIGURATION
# ============================================================================

def create_production_config() -> InformerConfig:
    """
    Create production configuration
    
    Modify this function based on your specific use case:
    - Energy forecasting: ConfigPresets.energy_forecasting()
    - Financial: ConfigPresets.financial_forecasting()
    - Weather: ConfigPresets.weather_forecasting()
    - Traffic: ConfigPresets.traffic_forecasting()
    """
    
    # Start with preset (or create from scratch)
    config = ConfigPresets.energy_forecasting()
    
    # ========================================
    # CUSTOMIZE FOR YOUR USE CASE
    # ========================================
    
    # Database configuration
    config.data_source.database_type = "postgresql"
    config.data_source.host = os.getenv("DB_HOST", "localhost")
    config.data_source.port = int(os.getenv("DB_PORT", 5432))
    config.data_source.database = os.getenv("DB_NAME", "timeseries_db")
    config.data_source.username = os.getenv("DB_USER", "user")
    config.data_source.password = os.getenv("DB_PASSWORD", "")
    config.data_source.use_ssl = True
    
    # Table schema
    config.data_schema.table_name = "your_table_name"
    config.data_schema.timestamp_column = "timestamp"
    config.data_schema.target_column = "target"
    config.data_schema.value_columns = [
        "feature_1",
        "feature_2", 
        "feature_3",
        "feature_4",
        "feature_5",
        "feature_6",
        "target"  # Include target in inputs for multivariate
    ]
    
    # Data filtering
    config.data_filter.start_date = "2023-01-01"
    config.data_filter.end_date = "2024-01-01"
    config.data_filter.exclude_nulls = True
    
    # Model architecture (adjust based on data complexity)
    config.architecture.enc_in = len(config.data_schema.value_columns)
    config.architecture.dec_in = len(config.data_schema.value_columns)
    config.architecture.c_out = 1
    config.architecture.seq_len = 96      # 96 hours lookback
    config.architecture.label_len = 48    # 48 hours start token
    config.architecture.pred_len = 24     # 24 hours forecast
    config.architecture.d_model = 512
    config.architecture.n_heads = 8
    config.architecture.e_layers = 2
    config.architecture.d_layers = 1
    config.architecture.freq = "h"
    
    # Training configuration
    config.training.batch_size = 32
    config.training.num_epochs = 10
    config.training.use_early_stopping = True
    config.training.patience = 3
    config.training.use_gpu = torch.cuda.is_available()
    config.training.num_workers = 4
    
    # Optimization
    config.optimization.learning_rate = 1e-4
    config.optimization.use_lr_scheduler = True
    config.optimization.use_mixed_precision = True
    
    # Paths
    config.training.checkpoint_dir = "./checkpoints/"
    config.training.tensorboard_dir = "./runs/"
    config.normalization.scaler_save_path = "./scalers/"
    
    # Experiment tracking
    config.experiment.experiment_name = "informer_production_v1"
    config.experiment.tracking_platform = "mlflow"
    config.experiment.tracking_uri = os.getenv("MLFLOW_URI", "http://localhost:5000")
    
    # Deployment
    config.deployment.environment = "production"
    config.deployment.model_version = "1.0.0"
    config.deployment.api_port = 8000
    config.deployment.enable_monitoring = True
    
    # Validate configuration
    config.validate()
    
    # Save configuration for reproducibility
    config.to_json("production_config.json")
    logger.info("✓ Configuration created and saved")
    
    return config


# ============================================================================
# STEP 2: DATA EXTRACTION
# ============================================================================

def extract_data(config: InformerConfig) -> pd.DataFrame:
    """
    Extract data from SQL database
    """
    logger.info("=" * 80)
    logger.info("STEP 2: DATA EXTRACTION")
    logger.info("=" * 80)
    
    # Create pipeline
    pipeline = SQLTimeSeriesPipeline(config)
    
    # Run extraction
    df = pipeline.run(
        validate_first=True,
        chunksize=50000  # Process in chunks for large datasets
    )
    
    # Save extracted data for faster iteration
    pipeline.save_to_parquet("data/raw_data.parquet")
    
    logger.info(f"✓ Extracted data shape: {df.shape}")
    logger.info(f"✓ Date range: {df[config.data_schema.timestamp_column].min()} to "
               f"{df[config.data_schema.timestamp_column].max()}")
    
    return df


# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================

def preprocess_data(
    df: pd.DataFrame,
    config: InformerConfig
) -> tuple:
    """
    Preprocess and split data
    """
    logger.info("=" * 80)
    logger.info("STEP 3: DATA PREPROCESSING")
    logger.info("=" * 80)
    
    # Import preprocessing modules
    # (These would be in your data_engineering_pipeline.py)
    from sklearn.preprocessing import StandardScaler
    
    # Sort by timestamp
    df = df.sort_values(config.data_schema.timestamp_column).reset_index(drop=True)
    
    # Handle missing values
    logger.info("Handling missing values...")
    for col in config.data_schema.value_columns:
        if col in df.columns:
            # Forward fill
            df[col] = df[col].fillna(method='ffill', limit=5)
            # Interpolate remaining
            df[col] = df[col].interpolate(method='linear', limit=3)
            # Fill any remaining with median
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
    
    # Extract temporal features
    logger.info("Extracting temporal features...")
    df['month'] = df[config.data_schema.timestamp_column].dt.month
    df['day'] = df[config.data_schema.timestamp_column].dt.day
    df['weekday'] = df[config.data_schema.timestamp_column].dt.dayofweek
    df['hour'] = df[config.data_schema.timestamp_column].dt.hour
    
    # Split data chronologically
    logger.info("Splitting data...")
    n = len(df)
    train_end = int(n * config.data_split.train_ratio)
    val_end = int(n * (config.data_split.train_ratio + config.data_split.val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Val:   {len(val_df)} samples")
    logger.info(f"Test:  {len(test_df)} samples")
    
    # Normalize (fit on train only!)
    logger.info("Normalizing data...")
    scaler = StandardScaler()
    
    value_cols = config.data_schema.value_columns
    train_df[value_cols] = scaler.fit_transform(train_df[value_cols])
    val_df[value_cols] = scaler.transform(val_df[value_cols])
    test_df[value_cols] = scaler.transform(test_df[value_cols])
    
    # Save scaler
    import joblib
    os.makedirs(config.normalization.scaler_save_path, exist_ok=True)
    scaler_path = os.path.join(
        config.normalization.scaler_save_path,
        config.normalization.scaler_filename
    )
    joblib.dump(scaler, scaler_path)
    logger.info(f"✓ Scaler saved to {scaler_path}")
    
    # Save preprocessed data
    os.makedirs("data", exist_ok=True)
    train_df.to_parquet("data/train.parquet", index=False)
    val_df.to_parquet("data/val.parquet", index=False)
    test_df.to_parquet("data/test.parquet", index=False)
    
    logger.info("✓ Preprocessing complete")
    
    return train_df, val_df, test_df, scaler


# ============================================================================
# STEP 4: MODEL TRAINING
# ============================================================================

def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: InformerConfig
):
    """
    Train Informer model
    
    NOTE: This is a skeleton. You need to implement the actual Informer model
    based on the architecture specs in model_specs.py
    """
    logger.info("=" * 80)
    logger.info("STEP 4: MODEL TRAINING")
    logger.info("=" * 80)
    
    # TODO: Implement Informer model (see IMPLEMENTATION_SPEC.md)
    # from models.informer import Informer, InformerDataset
    
    logger.info("Creating datasets...")
    # train_dataset = InformerDataset(train_df, config)
    # val_dataset = InformerDataset(val_df, config)
    
    logger.info("Creating data loaders...")
    # train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, ...)
    # val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, ...)
    
    logger.info("Initializing model...")
    # model = Informer(config.architecture)
    
    logger.info("Starting training...")
    # trainer = Trainer(model, config.training, config.optimization)
    # trainer.fit(train_loader, val_loader)
    
    logger.info("✓ Training complete")
    logger.info(f"✓ Best model saved to {config.training.checkpoint_dir}")


# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================

def evaluate_model(
    test_df: pd.DataFrame,
    config: InformerConfig,
    checkpoint_path: str
) -> Dict[str, float]:
    """
    Evaluate trained model on test set
    """
    logger.info("=" * 80)
    logger.info("STEP 5: MODEL EVALUATION")
    logger.info("=" * 80)
    
    # TODO: Implement evaluation
    # model = load_model(checkpoint_path)
    # test_dataset = InformerDataset(test_df, config)
    # test_loader = DataLoader(test_dataset, ...)
    
    # predictions, targets = predict(model, test_loader)
    
    # Calculate metrics
    metrics = {
        "mse": 0.0,  # np.mean((predictions - targets) ** 2)
        "mae": 0.0,  # np.mean(np.abs(predictions - targets))
        "rmse": 0.0, # np.sqrt(mse)
        "mape": 0.0  # np.mean(np.abs((predictions - targets) / targets)) * 100
    }
    
    logger.info("Test Set Metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name.upper()}: {metric_value:.4f}")
    
    # Save metrics
    import json
    with open("test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


# ============================================================================
# STEP 6: MODEL DEPLOYMENT
# ============================================================================

def deploy_model(config: InformerConfig, checkpoint_path: str):
    """
    Deploy model as REST API
    """
    logger.info("=" * 80)
    logger.info("STEP 6: MODEL DEPLOYMENT")
    logger.info("=" * 80)
    
    # Example FastAPI deployment
    """
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI(title="Informer Forecasting API")
    
    # Load model
    model = load_model(checkpoint_path)
    scaler = load_scaler(config.normalization.scaler_save_path)
    
    @app.post("/predict")
    async def predict(data: PredictionRequest):
        # Preprocess input
        # Run inference
        # Post-process output
        return {"predictions": predictions.tolist()}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    # Start server
    uvicorn.run(
        app,
        host=config.deployment.api_host,
        port=config.deployment.api_port,
        workers=config.deployment.api_workers
    )
    """
    
    logger.info(f"✓ API server configuration:")
    logger.info(f"  Host: {config.deployment.api_host}")
    logger.info(f"  Port: {config.deployment.api_port}")
    logger.info(f"  Workers: {config.deployment.api_workers}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_complete_pipeline():
    """
    Run complete end-to-end pipeline
    """
    logger.info("\n" + "=" * 80)
    logger.info("INFORMER MODEL DEPLOYMENT PIPELINE")
    logger.info("=" * 80 + "\n")
    
    try:
        # Step 1: Create configuration
        logger.info("STEP 1: CONFIGURATION")
        config = create_production_config()
        print(config)
        
        # Step 2: Extract data from SQL
        df = extract_data(config)
        
        # Step 3: Preprocess data
        train_df, val_df, test_df, scaler = preprocess_data(df, config)
        
        # Step 4: Train model
        train_model(train_df, val_df, config)
        
        # Step 5: Evaluate model
        checkpoint_path = os.path.join(
            config.training.checkpoint_dir,
            "best_model.pth"
        )
        metrics = evaluate_model(test_df, config, checkpoint_path)
        
        # Step 6: Deploy model
        deploy_model(config, checkpoint_path)
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_load_from_parquet():
    """
    Example: Skip SQL extraction and load from saved Parquet
    """
    logger.info("Loading data from Parquet...")
    
    train_df = pd.read_parquet("data/train.parquet")
    val_df = pd.read_parquet("data/val.parquet")
    test_df = pd.read_parquet("data/test.parquet")
    
    logger.info(f"Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df


def example_inference():
    """
    Example: Load model and run inference
    """
    logger.info("Running inference example...")
    
    # Load configuration
    config = InformerConfig.from_json("production_config.json")
    
    # Load model
    # model = load_model(config.inference.checkpoint_path)
    
    # Load scaler
    import joblib
    scaler_path = os.path.join(
        config.normalization.scaler_save_path,
        config.normalization.scaler_filename
    )
    scaler = joblib.load(scaler_path)
    
    # Prepare input data (last seq_len timesteps)
    # input_data = prepare_input(...)
    
    # Run inference
    # predictions = model.predict(input_data)
    
    # Denormalize predictions
    # predictions_original = scaler.inverse_transform(predictions)
    
    logger.info("✓ Inference complete")


def example_batch_prediction():
    """
    Example: Batch predictions for multiple entities
    """
    logger.info("Running batch prediction example...")
    
    config = InformerConfig.from_json("production_config.json")
    
    # List of entities to forecast
    entity_ids = ["ENTITY_001", "ENTITY_002", "ENTITY_003"]
    
    results = {}
    for entity_id in entity_ids:
        logger.info(f"Predicting for {entity_id}...")
        
        # Extract entity data
        # entity_data = extract_entity_data(entity_id, config)
        
        # Run prediction
        # predictions = predict(entity_data)
        
        # results[entity_id] = predictions
    
    logger.info(f"✓ Completed predictions for {len(entity_ids)} entities")


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Informer Model Deployment Pipeline"
    )
    
    parser.add_argument(
        "command",
        choices=[
            "full-pipeline",
            "extract-data",
            "train",
            "evaluate",
            "deploy",
            "inference"
        ],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="production_config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best_model.pth",
        help="Path to model checkpoint"
    )
    
    args = parser.parse_args()
    
    if args.command == "full-pipeline":
        run_complete_pipeline()
    
    elif args.command == "extract-data":
        config = InformerConfig.from_json(args.config)
        extract_data(config)
    
    elif args.command == "train":
        config = InformerConfig.from_json(args.config)
        train_df, val_df, _, _ = example_load_from_parquet()
        train_model(train_df, val_df, config)
    
    elif args.command == "evaluate":
        config = InformerConfig.from_json(args.config)
        _, _, test_df = example_load_from_parquet()
        evaluate_model(test_df, config, args.checkpoint)
    
    elif args.command == "deploy":
        config = InformerConfig.from_json(args.config)
        deploy_model(config, args.checkpoint)
    
    elif args.command == "inference":
        example_inference()
    
    else:
        parser.print_help()

