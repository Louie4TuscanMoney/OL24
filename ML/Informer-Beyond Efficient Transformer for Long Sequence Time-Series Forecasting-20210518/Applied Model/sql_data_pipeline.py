"""
SQL Data Pipeline for Informer Model
=====================================

Production-grade SQL data extraction and preprocessing pipeline
optimized for time series forecasting with Informer.

Supports: PostgreSQL, MySQL, BigQuery, Snowflake
Optimized for: Large-scale time series data (millions of rows)

Author: Engineering Team
Date: October 14, 2025
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta
import logging
from contextlib import contextmanager
from model_specs import (
    InformerConfig, DataSourceConfig, DataTableSchema,
    DataFilterConfig, DataPreprocessingConfig
)

# SQL Alchemy for database connections
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SQL QUERY BUILDER
# ============================================================================

class TimeSeriesQueryBuilder:
    """
    Build optimized SQL queries for time series data extraction
    """
    
    def __init__(self, config: InformerConfig):
        self.config = config
        self.schema = config.data_schema
        self.filters = config.data_filter
    
    def build_base_query(self) -> str:
        """
        Build base SELECT query for time series data
        
        Optimizations:
        - Select only necessary columns
        - Use indexed timestamp column
        - Apply filters at database level
        """
        # Columns to select
        columns = [
            self.schema.timestamp_column,
            *self.schema.value_columns,
            self.schema.target_column
        ]
        
        if self.schema.entity_column:
            columns.insert(0, self.schema.entity_column)
        
        columns.extend(self.schema.metadata_columns)
        
        # Remove duplicates while preserving order
        columns = list(dict.fromkeys(columns))
        
        column_list = ", ".join(columns)
        
        # Base query
        query = f"""
        SELECT {column_list}
        FROM {self.schema.table_name}
        WHERE {self.filters.to_sql_where_clause()}
        ORDER BY {self.schema.timestamp_column} ASC
        """
        
        return query
    
    def build_aggregated_query(
        self,
        agg_function: str = "AVG",
        time_bucket: str = "1 hour"
    ) -> str:
        """
        Build query with time-based aggregation
        Useful for downsampling high-frequency data
        
        Example: Aggregate minute data to hourly
        """
        value_aggs = ", ".join([
            f"{agg_function}({col}) as {col}"
            for col in self.schema.value_columns
        ])
        
        query = f"""
        SELECT 
            DATE_TRUNC('hour', {self.schema.timestamp_column}) as {self.schema.timestamp_column},
            {value_aggs},
            {agg_function}({self.schema.target_column}) as {self.schema.target_column}
        FROM {self.schema.table_name}
        WHERE {self.filters.to_sql_where_clause()}
        GROUP BY DATE_TRUNC('hour', {self.schema.timestamp_column})
        ORDER BY {self.schema.timestamp_column} ASC
        """
        
        return query
    
    def build_windowed_query(
        self,
        window_functions: Dict[str, str]
    ) -> str:
        """
        Build query with window functions for feature engineering
        
        Example:
        window_functions = {
            'lag_1': 'LAG(target, 1) OVER (ORDER BY timestamp)',
            'rolling_avg': 'AVG(target) OVER (ORDER BY timestamp ROWS BETWEEN 23 PRECEDING AND CURRENT ROW)'
        }
        """
        base_cols = ", ".join([
            self.schema.timestamp_column,
            *self.schema.value_columns,
            self.schema.target_column
        ])
        
        window_cols = ", ".join([
            f"{sql} as {name}"
            for name, sql in window_functions.items()
        ])
        
        query = f"""
        SELECT 
            {base_cols},
            {window_cols}
        FROM {self.schema.table_name}
        WHERE {self.filters.to_sql_where_clause()}
        ORDER BY {self.schema.timestamp_column} ASC
        """
        
        return query
    
    def build_multi_entity_query(self) -> str:
        """
        Build query for multi-entity time series
        Each entity has its own time series
        """
        if not self.schema.entity_column:
            raise ValueError("entity_column not configured")
        
        entity_filter = ""
        if self.filters.entity_ids:
            entity_list = ", ".join([f"'{e}'" for e in self.filters.entity_ids])
            entity_filter = f"AND {self.schema.entity_column} IN ({entity_list})"
        
        query = f"""
        SELECT *
        FROM {self.schema.table_name}
        WHERE {self.filters.to_sql_where_clause()}
        {entity_filter}
        ORDER BY {self.schema.entity_column}, {self.schema.timestamp_column} ASC
        """
        
        return query
    
    def build_data_quality_query(self) -> str:
        """
        Build query to check data quality before loading
        Returns summary statistics
        """
        query = f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT {self.schema.timestamp_column}) as unique_timestamps,
            MIN({self.schema.timestamp_column}) as min_timestamp,
            MAX({self.schema.timestamp_column}) as max_timestamp,
            COUNT(*) - COUNT({self.schema.target_column}) as null_target_count,
            AVG({self.schema.target_column}) as target_mean,
            STDDEV({self.schema.target_column}) as target_std,
            MIN({self.schema.target_column}) as target_min,
            MAX({self.schema.target_column}) as target_max
        FROM {self.schema.table_name}
        WHERE {self.filters.to_sql_where_clause()}
        """
        
        return query


# ============================================================================
# DATABASE CONNECTION MANAGER
# ============================================================================

class DatabaseConnection:
    """
    Manage database connections with connection pooling
    """
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.engine: Optional[Engine] = None
    
    def create_engine(self) -> Engine:
        """
        Create SQLAlchemy engine with connection pooling
        """
        connection_string = self.config.to_connection_string()
        
        engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=self.config.connection_pool_size,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=False  # Set True for SQL debugging
        )
        
        logger.info(f"Created database engine: {self.config.database_type}")
        return engine
    
    def get_engine(self) -> Engine:
        """Get or create database engine"""
        if self.engine is None:
            self.engine = self.create_engine()
        return self.engine
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections
        Ensures connections are properly closed
        """
        engine = self.get_engine()
        connection = engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            logger.info("✓ Database connection successful")
            return True
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            return False
    
    def close(self):
        """Close database engine"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine closed")


# ============================================================================
# DATA EXTRACTOR
# ============================================================================

class SQLDataExtractor:
    """
    Extract time series data from SQL database
    """
    
    def __init__(self, config: InformerConfig):
        self.config = config
        self.db_connection = DatabaseConnection(config.data_source)
        self.query_builder = TimeSeriesQueryBuilder(config)
    
    def extract_data(
        self,
        query: Optional[str] = None,
        chunksize: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract data from database
        
        Args:
            query: Custom SQL query (optional, uses default if None)
            chunksize: Process data in chunks (for large datasets)
        
        Returns:
            DataFrame with time series data
        """
        if query is None:
            query = self.query_builder.build_base_query()
        
        logger.info("Extracting data from database...")
        logger.debug(f"Query: {query}")
        
        try:
            if chunksize:
                # Process in chunks for large datasets
                chunks = []
                with self.db_connection.get_connection() as conn:
                    for chunk in pd.read_sql(
                        query,
                        conn,
                        chunksize=chunksize
                    ):
                        chunks.append(chunk)
                        logger.info(f"Loaded chunk: {len(chunk)} rows")
                
                df = pd.concat(chunks, ignore_index=True)
            else:
                # Load all at once
                with self.db_connection.get_connection() as conn:
                    df = pd.read_sql(query, conn)
            
            logger.info(f"✓ Extracted {len(df)} rows, {len(df.columns)} columns")
            return df
        
        except Exception as e:
            logger.error(f"✗ Data extraction failed: {e}")
            raise
    
    def check_data_quality(self) -> Dict[str, Any]:
        """
        Check data quality before extraction
        Returns summary statistics
        """
        query = self.query_builder.build_data_quality_query()
        
        try:
            with self.db_connection.get_connection() as conn:
                result = pd.read_sql(query, conn)
            
            quality_metrics = result.iloc[0].to_dict()
            
            logger.info("Data Quality Check:")
            for key, value in quality_metrics.items():
                logger.info(f"  {key}: {value}")
            
            return quality_metrics
        
        except Exception as e:
            logger.error(f"✗ Data quality check failed: {e}")
            raise
    
    def extract_by_entity(self, entity_id: str) -> pd.DataFrame:
        """
        Extract data for specific entity
        Useful for multi-entity forecasting
        """
        original_filter = self.config.data_filter.entity_ids
        self.config.data_filter.entity_ids = [entity_id]
        
        query = self.query_builder.build_multi_entity_query()
        df = self.extract_data(query)
        
        # Restore original filter
        self.config.data_filter.entity_ids = original_filter
        
        return df
    
    def extract_with_aggregation(
        self,
        agg_function: str = "AVG",
        time_bucket: str = "1 hour"
    ) -> pd.DataFrame:
        """
        Extract data with time-based aggregation
        Useful for downsampling
        """
        query = self.query_builder.build_aggregated_query(agg_function, time_bucket)
        return self.extract_data(query)
    
    def get_table_schema(self) -> Dict[str, str]:
        """
        Get database table schema
        Returns column names and data types
        """
        query = f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = '{self.config.data_schema.table_name}'
        ORDER BY ordinal_position
        """
        
        try:
            with self.db_connection.get_connection() as conn:
                schema_df = pd.read_sql(query, conn)
            
            schema = dict(zip(schema_df['column_name'], schema_df['data_type']))
            
            logger.info(f"Table schema ({len(schema)} columns):")
            for col, dtype in schema.items():
                logger.info(f"  {col}: {dtype}")
            
            return schema
        
        except Exception as e:
            logger.error(f"✗ Schema retrieval failed: {e}")
            raise


# ============================================================================
# DATA VALIDATOR
# ============================================================================

class SQLDataValidator:
    """
    Validate extracted data before preprocessing
    """
    
    def __init__(self, config: InformerConfig):
        self.config = config
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate extracted DataFrame
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required columns exist
        required_cols = [
            self.config.data_schema.timestamp_column,
            self.config.data_schema.target_column,
            *self.config.data_schema.value_columns
        ]
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check data types
        if self.config.data_schema.timestamp_column in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(
                df[self.config.data_schema.timestamp_column]
            ):
                issues.append("Timestamp column is not datetime type")
        
        # Check for empty data
        if len(df) == 0:
            issues.append("DataFrame is empty")
        
        # Check for duplicate timestamps
        if self.config.data_schema.timestamp_column in df.columns:
            duplicates = df[self.config.data_schema.timestamp_column].duplicated().sum()
            if duplicates > 0:
                issues.append(f"Found {duplicates} duplicate timestamps")
        
        # Check missing values
        for col in self.config.data_schema.value_columns:
            if col in df.columns:
                missing_pct = df[col].isna().sum() / len(df)
                if missing_pct > self.config.preprocessing.missing_threshold:
                    issues.append(
                        f"Column '{col}' has {missing_pct:.1%} missing values "
                        f"(threshold: {self.config.preprocessing.missing_threshold:.1%})"
                    )
        
        # Check timestamp ordering
        if self.config.data_schema.timestamp_column in df.columns:
            if not df[self.config.data_schema.timestamp_column].is_monotonic_increasing:
                issues.append("Timestamps are not in chronological order")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("✓ Data validation passed")
        else:
            logger.warning(f"✗ Data validation failed with {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues


# ============================================================================
# COMPLETE DATA PIPELINE
# ============================================================================

class SQLTimeSeriesPipeline:
    """
    Complete end-to-end pipeline for SQL data extraction and preprocessing
    """
    
    def __init__(self, config: InformerConfig):
        self.config = config
        self.extractor = SQLDataExtractor(config)
        self.validator = SQLDataValidator(config)
        self.data: Optional[pd.DataFrame] = None
    
    def run(
        self,
        validate_first: bool = True,
        chunksize: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Run complete data pipeline
        
        Steps:
        1. Test database connection
        2. Check data quality
        3. Extract data
        4. Validate data
        5. Convert data types
        
        Returns:
            Cleaned DataFrame ready for preprocessing
        """
        logger.info("=" * 80)
        logger.info("STARTING SQL TIME SERIES DATA PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Test connection
        logger.info("\n[1/5] Testing database connection...")
        if not self.extractor.db_connection.test_connection():
            raise ConnectionError("Database connection failed")
        
        # Step 2: Check data quality (optional)
        if validate_first:
            logger.info("\n[2/5] Checking data quality...")
            quality_metrics = self.extractor.check_data_quality()
            
            # Validate quality metrics
            if quality_metrics['total_rows'] == 0:
                raise ValueError("No data found in database")
            
            if quality_metrics['null_target_count'] / quality_metrics['total_rows'] > 0.5:
                logger.warning("Warning: >50% of target values are NULL")
        
        # Step 3: Extract data
        logger.info("\n[3/5] Extracting data from database...")
        df = self.extractor.extract_data(chunksize=chunksize)
        
        # Step 4: Validate extracted data
        logger.info("\n[4/5] Validating extracted data...")
        is_valid, issues = self.validator.validate_dataframe(df)
        
        if not is_valid:
            logger.error("Data validation failed!")
            for issue in issues:
                logger.error(f"  - {issue}")
            raise ValueError(f"Data validation failed with {len(issues)} issues")
        
        # Step 5: Convert data types
        logger.info("\n[5/5] Converting data types...")
        df = self._convert_dtypes(df)
        
        self.data = df
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
        logger.info(f"Date range: {df[self.config.data_schema.timestamp_column].min()} to "
                   f"{df[self.config.data_schema.timestamp_column].max()}")
        
        return df
    
    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame columns to appropriate data types"""
        # Convert timestamp to datetime
        timestamp_col = self.config.data_schema.timestamp_column
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Convert value columns to float32
        value_cols = [
            *self.config.data_schema.value_columns,
            self.config.data_schema.target_column
        ]
        
        for col in value_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
        
        logger.info("✓ Data types converted")
        return df
    
    def save_to_parquet(self, filepath: str):
        """Save extracted data to Parquet file for faster loading"""
        if self.data is None:
            raise ValueError("No data to save. Run pipeline first.")
        
        self.data.to_parquet(
            filepath,
            compression='snappy',
            engine='pyarrow',
            index=False
        )
        logger.info(f"✓ Data saved to {filepath}")
    
    def get_data_summary(self) -> pd.DataFrame:
        """Get summary statistics of extracted data"""
        if self.data is None:
            raise ValueError("No data available. Run pipeline first.")
        
        summary = self.data.describe()
        return summary


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_basic_usage():
    """Example: Basic data extraction"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Data Extraction")
    print("=" * 80)
    
    # Create configuration
    from model_specs import ConfigPresets
    config = ConfigPresets.energy_forecasting()
    
    # Configure database connection
    config.data_source.host = "your-database-host.com"
    config.data_source.database = "timeseries_production"
    config.data_source.username = "readonly_user"
    # Password should be set via environment variable
    
    # Configure data table
    config.data_schema.table_name = "energy_loads"
    config.data_schema.timestamp_column = "measured_at"
    config.data_schema.target_column = "load_mw"
    config.data_schema.value_columns = ["temperature", "humidity", "wind_speed"]
    
    # Set date range
    config.data_filter.start_date = "2023-01-01"
    config.data_filter.end_date = "2023-12-31"
    
    # Run pipeline
    pipeline = SQLTimeSeriesPipeline(config)
    df = pipeline.run()
    
    print("\nData preview:")
    print(df.head())
    
    # Save for later use
    pipeline.save_to_parquet("energy_data.parquet")


def example_multi_entity():
    """Example: Multi-entity forecasting"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multi-Entity Forecasting")
    print("=" * 80)
    
    from model_specs import ConfigPresets
    config = ConfigPresets.traffic_forecasting()
    
    # Configure for multiple sensors
    config.data_schema.entity_column = "sensor_id"
    config.data_filter.entity_ids = ["SENSOR_001", "SENSOR_002", "SENSOR_003"]
    
    # Extract data
    pipeline = SQLTimeSeriesPipeline(config)
    df = pipeline.run()
    
    print(f"\nExtracted data for {df['sensor_id'].nunique()} sensors")
    print(df.groupby('sensor_id').size())


def example_custom_query():
    """Example: Custom SQL query"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom SQL Query")
    print("=" * 80)
    
    from model_specs import InformerConfig
    config = InformerConfig()
    
    # Custom query with JOIN
    custom_query = """
    SELECT 
        t.timestamp,
        t.value,
        w.temperature,
        w.humidity
    FROM timeseries_data t
    LEFT JOIN weather_data w
        ON DATE(t.timestamp) = DATE(w.timestamp)
    WHERE t.timestamp >= '2023-01-01'
    ORDER BY t.timestamp
    """
    
    extractor = SQLDataExtractor(config)
    df = extractor.extract_data(query=custom_query)
    
    print(f"\nExtracted {len(df)} rows with custom query")


if __name__ == "__main__":
    print("SQL Data Pipeline for Informer Model")
    print("=" * 80)
    print("\nThis module provides production-grade SQL data extraction.")
    print("See example functions for usage patterns.")
    print("\nAvailable examples:")
    print("  - example_basic_usage()")
    print("  - example_multi_entity()")
    print("  - example_custom_query()")

