"""
Informer Model Specifications for Production Deployment
========================================================

Silicon Valley Data Science Startup - Time Series Forecasting Platform

This module contains all configuration classes, hyperparameters, and
specifications for deploying Informer models at scale.

Author: Engineering Team
Date: October 14, 2025
Version: 1.0.0
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import json


# ============================================================================
# ENUMERATIONS
# ============================================================================

class ModelType(Enum):
    """Supported model architectures"""
    INFORMER = "informer"
    INFORMER_STACK = "informerstack"
    VANILLA_TRANSFORMER = "transformer"


class AttentionType(Enum):
    """Attention mechanism types"""
    PROB_SPARSE = "prob"
    FULL = "full"


class DataFrequency(Enum):
    """Time series data frequencies"""
    HOURLY = "h"
    MINUTE_15 = "t"  # 15-minute intervals
    DAILY = "d"
    WEEKLY = "w"
    MONTHLY = "m"


class FeatureType(Enum):
    """Feature configuration types"""
    MULTIVARIATE = "M"  # Multiple input, multiple output
    UNIVARIATE = "S"   # Single input, single output
    MULTIVARIATE_TO_UNIVARIATE = "MS"  # Multiple input, single output


class LossFunction(Enum):
    """Supported loss functions"""
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    SMAPE = "smape"


class OptimizationType(Enum):
    """Optimizer types"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class EnvironmentType(Enum):
    """Deployment environments"""
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

@dataclass
class DataSourceConfig:
    """Configuration for SQL data source connection"""
    
    # Database connection
    database_type: str = "postgresql"  # postgresql, mysql, bigquery, snowflake
    host: str = "localhost"
    port: int = 5432
    database: str = "timeseries_db"
    schema: str = "public"
    
    # Authentication
    username: str = ""
    password: str = ""  # Use environment variable in production
    connection_pool_size: int = 5
    
    # SSL/Security
    use_ssl: bool = True
    ssl_cert_path: Optional[str] = None
    
    # Query optimization
    fetch_size: int = 10000
    timeout_seconds: int = 300
    
    def to_connection_string(self) -> str:
        """Generate database connection string"""
        if self.database_type == "postgresql":
            return (f"postgresql://{self.username}:{self.password}@"
                   f"{self.host}:{self.port}/{self.database}")
        elif self.database_type == "mysql":
            return (f"mysql+pymysql://{self.username}:{self.password}@"
                   f"{self.host}:{self.port}/{self.database}")
        elif self.database_type == "bigquery":
            return f"bigquery://{self.database}"
        else:
            raise ValueError(f"Unsupported database: {self.database_type}")


@dataclass
class DataTableSchema:
    """Schema definition for time series data table"""
    
    # Table information
    table_name: str = "time_series_data"
    timestamp_column: str = "timestamp"
    
    # Feature columns
    value_columns: List[str] = field(default_factory=lambda: [
        "value_1", "value_2", "value_3"
    ])
    target_column: str = "target"
    
    # Optional columns
    entity_column: Optional[str] = None  # For multi-entity forecasting
    metadata_columns: List[str] = field(default_factory=list)
    
    # Data types
    timestamp_dtype: str = "timestamp"
    value_dtype: str = "float32"
    
    # Partitioning (for large datasets)
    partition_column: Optional[str] = "date"
    partition_type: Optional[str] = "range"  # range, hash, list


@dataclass
class DataFilterConfig:
    """SQL query filters and constraints"""
    
    # Time range
    start_date: Optional[str] = None  # ISO format: "2023-01-01"
    end_date: Optional[str] = None
    
    # Entity filtering
    entity_ids: Optional[List[str]] = None
    entity_filter_sql: Optional[str] = None
    
    # Data quality filters
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    exclude_nulls: bool = True
    exclude_outliers: bool = False
    outlier_std_threshold: float = 4.0
    
    # Sampling (for large datasets)
    sample_rate: Optional[float] = None  # 0.0 to 1.0
    sample_method: str = "random"  # random, systematic
    
    def to_sql_where_clause(self) -> str:
        """Generate SQL WHERE clause from filters"""
        conditions = []
        
        if self.start_date:
            conditions.append(f"timestamp >= '{self.start_date}'")
        if self.end_date:
            conditions.append(f"timestamp <= '{self.end_date}'")
        if self.entity_filter_sql:
            conditions.append(self.entity_filter_sql)
        if self.exclude_nulls:
            conditions.append("target IS NOT NULL")
        
        return " AND ".join(conditions) if conditions else "1=1"


@dataclass
class DataPreprocessingConfig:
    """Data cleaning and preprocessing parameters"""
    
    # Missing value handling
    missing_value_strategy: str = "forward_fill"  # forward_fill, interpolate, median
    max_consecutive_missing: int = 5
    missing_threshold: float = 0.1  # Drop series if >10% missing
    
    # Outlier detection
    outlier_detection_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_treatment: str = "clip"  # clip, remove, interpolate
    iqr_multiplier: float = 3.0
    zscore_threshold: float = 4.0
    
    # Timestamp regularization
    regularize_timestamps: bool = True
    expected_frequency: str = "1H"
    frequency_tolerance: str = "5min"
    
    # Data validation
    check_stationarity: bool = True
    check_seasonality: bool = True
    auto_difference: bool = False
    
    # Feature engineering
    create_time_features: bool = True
    create_lag_features: bool = False
    lag_periods: List[int] = field(default_factory=lambda: [1, 7, 24])
    create_rolling_features: bool = False
    rolling_windows: List[int] = field(default_factory=lambda: [7, 24])


@dataclass
class NormalizationConfig:
    """Normalization/scaling configuration"""
    
    # Normalization method
    method: str = "standard"  # standard, minmax, robust, maxabs
    per_feature: bool = True
    
    # Fit on train only
    fit_on_train_only: bool = True
    
    # Clipping (after normalization)
    clip_values: bool = False
    clip_min: float = -10.0
    clip_max: float = 10.0
    
    # Save/load scalers
    scaler_save_path: str = "./scalers/"
    scaler_filename: str = "scaler.pkl"


# ============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# ============================================================================

@dataclass
class InformerArchitectureConfig:
    """Informer model architecture hyperparameters"""
    
    # Model type
    model_type: ModelType = ModelType.INFORMER
    attention_type: AttentionType = AttentionType.PROB_SPARSE
    
    # Input/Output dimensions
    enc_in: int = 7  # Number of encoder input features
    dec_in: int = 7  # Number of decoder input features
    c_out: int = 1   # Number of output features
    
    # Sequence lengths
    seq_len: int = 96     # Input sequence length
    label_len: int = 48   # Start token length for decoder
    pred_len: int = 24    # Prediction horizon
    
    # Model dimensions
    d_model: int = 512    # Model dimension
    d_ff: int = 2048      # Feed-forward dimension
    n_heads: int = 8      # Number of attention heads
    
    # Layer configuration
    e_layers: int = 2     # Number of encoder layers
    d_layers: int = 1     # Number of decoder layers
    
    # ProbSparse attention
    factor: int = 5       # Sampling factor for ProbSparse
    
    # Distilling
    distil: bool = True   # Use attention distilling
    
    # Activation and dropout
    activation: str = "gelu"  # gelu, relu
    dropout: float = 0.05
    
    # Embedding
    embed_type: str = "timeF"  # timeF, fixed, learned
    freq: str = "h"  # Data frequency for time features
    
    # Output attention (for visualization)
    output_attention: bool = False
    
    def validate(self):
        """Validate architecture configuration"""
        assert self.seq_len > 0, "seq_len must be positive"
        assert self.pred_len > 0, "pred_len must be positive"
        assert self.label_len > 0, "label_len must be positive"
        assert self.label_len <= self.seq_len, "label_len cannot exceed seq_len"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.e_layers > 0, "Must have at least 1 encoder layer"
        assert self.d_layers > 0, "Must have at least 1 decoder layer"


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class OptimizationConfig:
    """Training optimization parameters"""
    
    # Optimizer
    optimizer: OptimizationType = OptimizationType.ADAM
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    scheduler_type: str = "cosine"  # cosine, step, exponential, plateau
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    
    # Gradient clipping
    use_grad_clip: bool = True
    max_grad_norm: float = 1.0
    
    # Mixed precision training
    use_mixed_precision: bool = True
    
    # Loss function
    loss_function: LossFunction = LossFunction.MSE
    
    # Regularization
    label_smoothing: float = 0.0


@dataclass
class TrainingConfig:
    """Training loop configuration"""
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 10
    
    # Early stopping
    use_early_stopping: bool = True
    patience: int = 3
    min_delta: float = 1e-4
    
    # Validation
    val_check_interval: int = 1  # Validate every N epochs
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints/"
    save_best_only: bool = True
    checkpoint_metric: str = "val_loss"  # val_loss, val_mae, val_mse
    
    # Logging
    log_interval: int = 100  # Log every N batches
    use_tensorboard: bool = True
    tensorboard_dir: str = "./runs/"
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    
    # Hardware
    use_gpu: bool = True
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    use_multi_gpu: bool = False
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class DataSplitConfig:
    """Train/validation/test split configuration"""
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # Split method
    split_method: str = "chronological"  # chronological, random (don't use random!)
    
    # Validation strategy
    use_time_series_cv: bool = False
    cv_folds: int = 5
    cv_gap: int = 0  # Gap between train and validation
    
    def validate(self):
        """Validate split configuration"""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {total}"
        assert self.split_method == "chronological", \
            "Time series must use chronological split!"


# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

@dataclass
class InferenceConfig:
    """Model inference configuration"""
    
    # Model loading
    checkpoint_path: str = "./checkpoints/best_model.pth"
    device: str = "cuda"  # cuda, cpu
    
    # Batch inference
    inference_batch_size: int = 64
    
    # Prediction settings
    return_confidence_intervals: bool = False
    confidence_level: float = 0.95
    num_samples: int = 100  # For uncertainty estimation
    
    # Output format
    denormalize_output: bool = True
    output_format: str = "dataframe"  # dataframe, array, dict
    
    # Performance
    use_jit_compilation: bool = False
    use_onnx: bool = False
    onnx_model_path: Optional[str] = None


# ============================================================================
# DEPLOYMENT CONFIGURATION
# ============================================================================

@dataclass
class DeploymentConfig:
    """Production deployment configuration"""
    
    # Environment
    environment: EnvironmentType = EnvironmentType.PRODUCTION
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Model serving
    model_version: str = "1.0.0"
    model_registry_path: str = "./model_registry/"
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_endpoint: str = "/metrics"
    log_predictions: bool = True
    
    # Performance
    max_concurrent_requests: int = 100
    request_timeout_seconds: int = 30
    
    # Caching
    use_redis_cache: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl_seconds: int = 3600
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_utilization: float = 0.7


# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

@dataclass
class ExperimentConfig:
    """MLOps experiment tracking configuration"""
    
    # Experiment identification
    experiment_name: str = "informer_baseline"
    run_name: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Tracking platform
    tracking_platform: str = "mlflow"  # mlflow, wandb, tensorboard
    tracking_uri: str = "http://localhost:5000"
    
    # Artifact logging
    log_model: bool = True
    log_code: bool = True
    log_data_samples: bool = False
    
    # Metrics to track
    primary_metric: str = "val_mse"
    secondary_metrics: List[str] = field(default_factory=lambda: [
        "val_mae", "val_rmse", "train_loss"
    ])


# ============================================================================
# COMPLETE CONFIGURATION MANAGER
# ============================================================================

@dataclass
class InformerConfig:
    """
    Complete configuration for Informer model pipeline
    
    This is the master configuration class that combines all sub-configurations.
    """
    
    # Sub-configurations
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    data_schema: DataTableSchema = field(default_factory=DataTableSchema)
    data_filter: DataFilterConfig = field(default_factory=DataFilterConfig)
    preprocessing: DataPreprocessingConfig = field(default_factory=DataPreprocessingConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    
    architecture: InformerArchitectureConfig = field(default_factory=InformerArchitectureConfig)
    
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data_split: DataSplitConfig = field(default_factory=DataSplitConfig)
    
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Metadata
    config_version: str = "1.0.0"
    created_at: Optional[str] = None
    description: Optional[str] = None
    
    def validate(self):
        """Validate entire configuration"""
        self.architecture.validate()
        self.data_split.validate()
        
        # Cross-validation between configs
        assert self.architecture.enc_in == len(self.data_schema.value_columns), \
            "enc_in must match number of value columns"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def from_json(cls, filepath: str) -> 'InformerConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested dataclasses
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        """Pretty print configuration"""
        lines = ["=" * 80]
        lines.append("INFORMER MODEL CONFIGURATION")
        lines.append("=" * 80)
        
        lines.append("\n[DATA SOURCE]")
        lines.append(f"  Database: {self.data_source.database_type}")
        lines.append(f"  Table: {self.data_schema.table_name}")
        
        lines.append("\n[ARCHITECTURE]")
        lines.append(f"  Model: {self.architecture.model_type.value}")
        lines.append(f"  Sequence: {self.architecture.seq_len} → {self.architecture.pred_len}")
        lines.append(f"  Dimensions: d_model={self.architecture.d_model}, heads={self.architecture.n_heads}")
        
        lines.append("\n[TRAINING]")
        lines.append(f"  Batch size: {self.training.batch_size}")
        lines.append(f"  Epochs: {self.training.num_epochs}")
        lines.append(f"  Learning rate: {self.optimization.learning_rate}")
        
        lines.append("\n[DEPLOYMENT]")
        lines.append(f"  Environment: {self.deployment.environment.value}")
        lines.append(f"  API: {self.deployment.api_host}:{self.deployment.api_port}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

class ConfigPresets:
    """Pre-configured setups for common use cases"""
    
    @staticmethod
    def small_dataset() -> InformerConfig:
        """Configuration for small datasets (<100K samples)"""
        config = InformerConfig()
        config.architecture.d_model = 256
        config.architecture.d_ff = 1024
        config.architecture.n_heads = 4
        config.architecture.e_layers = 2
        config.training.batch_size = 64
        config.training.num_epochs = 20
        return config
    
    @staticmethod
    def medium_dataset() -> InformerConfig:
        """Configuration for medium datasets (100K-1M samples)"""
        config = InformerConfig()
        config.architecture.d_model = 512
        config.architecture.d_ff = 2048
        config.architecture.n_heads = 8
        config.architecture.e_layers = 3
        config.training.batch_size = 32
        config.training.num_epochs = 10
        return config
    
    @staticmethod
    def large_dataset() -> InformerConfig:
        """Configuration for large datasets (>1M samples)"""
        config = InformerConfig()
        config.architecture.d_model = 512
        config.architecture.d_ff = 2048
        config.architecture.n_heads = 8
        config.architecture.e_layers = 2
        config.training.batch_size = 16
        config.training.num_epochs = 5
        config.training.use_multi_gpu = True
        return config
    
    @staticmethod
    def energy_forecasting() -> InformerConfig:
        """Preset for energy/electricity load forecasting"""
        config = InformerConfig()
        config.data_schema.table_name = "energy_load"
        config.data_schema.target_column = "load_mw"
        config.architecture.seq_len = 168  # 1 week hourly
        config.architecture.label_len = 84
        config.architecture.pred_len = 24  # 1 day ahead
        config.architecture.freq = "h"
        config.preprocessing.create_time_features = True
        return config
    
    @staticmethod
    def financial_forecasting() -> InformerConfig:
        """Preset for financial/stock price forecasting"""
        config = InformerConfig()
        config.data_schema.table_name = "stock_prices"
        config.data_schema.target_column = "close_price"
        config.architecture.seq_len = 252  # 1 year trading days
        config.architecture.label_len = 126
        config.architecture.pred_len = 20  # 1 month ahead
        config.architecture.freq = "d"
        config.normalization.method = "robust"  # Better for financial data
        return config
    
    @staticmethod
    def weather_forecasting() -> InformerConfig:
        """Preset for weather/temperature forecasting"""
        config = InformerConfig()
        config.data_schema.table_name = "weather_data"
        config.data_schema.target_column = "temperature"
        config.architecture.seq_len = 144  # 24 hours (10-min intervals)
        config.architecture.label_len = 72
        config.architecture.pred_len = 48  # 8 hours ahead
        config.architecture.freq = "t"
        return config
    
    @staticmethod
    def traffic_forecasting() -> InformerConfig:
        """Preset for traffic flow forecasting"""
        config = InformerConfig()
        config.data_schema.table_name = "traffic_flow"
        config.data_schema.target_column = "flow_rate"
        config.architecture.seq_len = 96  # 24 hours (15-min intervals)
        config.architecture.label_len = 48
        config.architecture.pred_len = 12  # 3 hours ahead
        config.architecture.freq = "t"
        return config


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_config_from_preset(preset_name: str) -> InformerConfig:
    """
    Create configuration from preset name
    
    Args:
        preset_name: One of 'small', 'medium', 'large', 'energy', 'financial', 'weather', 'traffic'
    
    Returns:
        InformerConfig instance
    """
    presets = {
        'small': ConfigPresets.small_dataset,
        'medium': ConfigPresets.medium_dataset,
        'large': ConfigPresets.large_dataset,
        'energy': ConfigPresets.energy_forecasting,
        'financial': ConfigPresets.financial_forecasting,
        'weather': ConfigPresets.weather_forecasting,
        'traffic': ConfigPresets.traffic_forecasting,
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
    
    return presets[preset_name]()


def merge_configs(base_config: InformerConfig, override_config: Dict[str, Any]) -> InformerConfig:
    """
    Merge base configuration with override dictionary
    
    Args:
        base_config: Base configuration
        override_config: Dictionary with override values
    
    Returns:
        Merged configuration
    """
    config_dict = base_config.to_dict()
    
    # Deep merge
    def deep_merge(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and key in base:
                base[key] = deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    merged = deep_merge(config_dict, override_config)
    return InformerConfig(**merged)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Create default configuration
    print("=" * 80)
    print("EXAMPLE 1: Default Configuration")
    print("=" * 80)
    config = InformerConfig()
    print(config)
    
    # Example 2: Create from preset
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Energy Forecasting Preset")
    print("=" * 80)
    energy_config = ConfigPresets.energy_forecasting()
    print(energy_config)
    
    # Example 3: Save and load configuration
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Save/Load Configuration")
    print("=" * 80)
    energy_config.to_json("example_config.json")
    loaded_config = InformerConfig.from_json("example_config.json")
    print("Configuration loaded successfully!")
    
    # Example 4: Customize configuration
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Custom Configuration")
    print("=" * 80)
    custom_config = ConfigPresets.medium_dataset()
    custom_config.data_source.host = "production-db.company.com"
    custom_config.data_schema.table_name = "custom_timeseries"
    custom_config.architecture.seq_len = 200
    custom_config.training.batch_size = 64
    print(custom_config)
    
    # Validate
    custom_config.validate()
    print("\n✓ Configuration validated successfully!")

